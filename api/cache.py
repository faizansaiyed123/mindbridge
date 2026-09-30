"""Query cache: in-process LRU -> Redis exact key -> Redis semantic match.

Three layers, tried in that order, each strictly cheaper than the one below:

1. **LRU** (in-process, bounded). The identical query from the same process.
   No network at all.
2. **Exact key** (Redis). The identical query from any process. The key is a
   blake2b hash of the normalised query text plus its parameters.
3. **Semantic** (Redis). A *different* query whose embedding is near a cached
   one. This is the M5 layer and it is the only one that can be wrong.

Layers 1 and 2 cannot return another question's answer: the key is a hash of
the question. Layer 3 can, and that is the error this module is designed
against, because a false cache hit does not look like a failure — it returns a
confident, well-formed answer built from some other question's memories.

Five defences, in order of how much they actually buy:

- **Parameters are a hard partition, never a similarity.** Only the query text
  is matched by embedding. `top_k`, the time window, the category filter and
  `include_superseded` are hashed into a fingerprint that must match byte for
  byte. A cached top_k=3 result can never be served to a top_k=5 request, no
  matter how identical the question.
- **Nearest neighbour only, above an absolute threshold.** One candidate is
  ever eligible — the argmax — and only if it clears
  `cache_semantic_threshold`. Second place is never served.
- **An ambiguity margin.** If the runner-up is within `cache_semantic_margin`
  of the winner, the lookup refuses. This exists because of a measured failure
  mode, not a hypothetical one: under nomic-embed-text several *unrelated*
  short Chinese queries collapse to cosine ~1.0 of each other. When that
  happens the winner is arbitrary, so the margin converts the collapse into a
  cache miss instead of a wrong answer.
- **A write invalidates the semantic index with everything else.** The index
  lives under the same `mindbridge:{namespace}:*` prefix the namespace flush
  already sweeps, so a new preference cannot be shadowed by a stale
  neighbour.
- **Off unless measured.** `cache_semantic_enabled` defaults to False. The
  threshold is embedder-specific and, under the embedder this project ships
  with, no threshold reaches a zero false-hit rate. See the `cache_semantic`
  measurement in evals/results.json.

The asymmetry that drives all of it: a missed cache hit costs one vector
search. A false cache hit answers the wrong question and nobody notices.

中文说明：三层缓存按开销依次尝试：进程内 LRU、Redis 精确键、Redis 语义
匹配。前两层只命中相同问题；第三层可能把答案错配给另一个问题，因此有五道
防线：参数硬分区、最近邻阈值、歧义边界、写入失效和默认关闭。漏掉命中只会
多做一次检索，错误命中却会返回看似完整、实际答非所问的结果。
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Any, NamedTuple

import redis.asyncio as redis

from .settings import Settings

logger = logging.getLogger(__name__)

_SEMANTIC_INDEX_SUFFIX = "semindex"


class SemanticLookup(NamedTuple):
    """Why a semantic lookup did what it did — every field is for the eval.

    中文：记录一次语义查找为何命中或拒绝；所有字段都服务于评测。
    """

    value: dict[str, Any] | None
    best_similarity: float | None
    runner_up_similarity: float | None
    matched_query: str | None
    candidates: int
    # served | below_threshold | ambiguous | no_candidates | disabled
    outcome: str


class _LruCache:
    """Bounded, TTL-aware, in-process. Sits in front of Redis.

    中文：有容量上限、带 TTL 的进程内缓存，位于 Redis 之前。
    """

    def __init__(self, capacity: int, ttl_seconds: int) -> None:
        """Set the capacity and TTL; entries are created lazily.

        中文：设置容量和过期时间；条目在首次写入时创建。
        """
        self._capacity = capacity
        self._ttl = ttl_seconds
        self._entries: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()

    def get(self, key: str) -> dict[str, Any] | None:
        """Return the cached value for key, or None if absent/expired.

        中文：返回 key 对应的值；不存在或已过期时返回 None。
        """
        if self._capacity <= 0:
            return None
        entry = self._entries.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at <= time.monotonic():
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return value

    def set(self, key: str, value: dict[str, Any]) -> None:
        """Store a value with a fresh TTL and enforce the capacity bound.

        中文：以新的过期时间保存值；超出容量时淘汰最久未使用的条目。
        """
        if self._capacity <= 0:
            return
        self._entries[key] = (time.monotonic() + self._ttl, value)
        self._entries.move_to_end(key)
        while len(self._entries) > self._capacity:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        """Drop every entry.

        中文：清空所有缓存条目。
        """
        self._entries.clear()


def _dot(left: list[float], right: list[float]) -> float:
    """Return the dot product of two equal-length vectors.

    中文：两个向量都经过 L2 归一化时，点积等于余弦相似度。
    """
    return sum(a * b for a, b in zip(left, right))


class QueryCache:
    """The three-layer temporal_query cache described in the module docstring.

    中文：即模块文档所述的三层 temporal_query 缓存(LRU -> Redis 精确键 ->
    Redis 语义匹配)。
    """

    def __init__(
        self,
        client: redis.Redis | None,
        ttl_seconds: int,
        *,
        lru_size: int = 128,
        semantic_enabled: bool = False,
        semantic_threshold: float = 0.95,
        semantic_margin: float = 0.02,
        semantic_index_size: int = 256,
    ) -> None:
        """Configure the LRU and semantic matching, then zero the counters.

        中文：配置 LRU 与语义匹配参数，并初始化统计计数器。
        """
        self._client = client
        self._ttl = ttl_seconds
        self._lru = _LruCache(lru_size, ttl_seconds)
        self.semantic_enabled = semantic_enabled
        self.semantic_threshold = semantic_threshold
        self.semantic_margin = semantic_margin
        self._semantic_index_size = semantic_index_size
        # Counters, split by layer so a saving can be attributed rather than
        # asserted. A semantic hit and an exact hit do not save the same work:
        # the exact path skips embedding *and* retrieval, the semantic path has
        # already paid for the embedding by the time it can look anything up.
        self.lru_hits = 0
        self.hits = 0
        self.semantic_hits = 0
        self.semantic_rejections = 0
        self.misses = 0

    @classmethod
    async def connect(cls, settings: Settings) -> "QueryCache":
        """Connect to Redis if enabled and reachable, else fall back to LRU-only.

        中文：缓存启用且 Redis 可达时建立连接；否则退化为仅使用进程内
        LRU。

        Args:
            settings: Redis, TTL, LRU, and semantic-cache configuration.

        Returns:
            A Redis-backed cache, or an LRU-only fallback.
        """
        options: dict[str, Any] = {
            "lru_size": settings.cache_lru_size,
            "semantic_enabled": settings.cache_semantic_enabled,
            "semantic_threshold": settings.cache_semantic_threshold,
            "semantic_margin": settings.cache_semantic_margin,
            "semantic_index_size": settings.cache_semantic_index_size,
        }
        if not settings.cache_enabled:
            return cls(None, settings.cache_ttl_seconds, **options)
        client: redis.Redis = redis.from_url(
            str(settings.redis_url), decode_responses=True
        )
        try:
            await client.ping()
        except Exception as error:  # pragma: no cover - depends on deployment
            logger.warning("redis unavailable (%s); running without cache", error)
            await client.aclose()
            return cls(None, settings.cache_ttl_seconds, **options)
        return cls(client, settings.cache_ttl_seconds, **options)

    async def health(self) -> dict[str, str]:
        """Probe the Redis dependency used by this cache.

        Returns ``disabled`` when the cache is intentionally configured without
        Redis, otherwise performs a real Redis PING so /healthz reflects the
        dependency that the API actually relies on.
        """
        if self._client is None:
            return {"status": "disabled"}
        try:
            await self._client.ping()
        except Exception as error:
            logger.warning("redis health check failed: %s", type(error).__name__)
            return {"status": "error", "detail": "Redis PING failed"}
        return {"status": "ok"}

    @property
    def enabled(self) -> bool:
        """Return whether a live Redis client is attached.

        中文：返回当前是否连接了可用的 Redis 客户端。
        """
        return self._client is not None

    # --- keys -------------------------------------------------------------

    @staticmethod
    def key(namespace: str, payload: dict[str, Any]) -> str:
        """Build the exact-match cache key: a namespaced hash of the whole payload.

        中文：对完整 payload（含查询文本与参数）做哈希，并加命名空间
        前缀。

        Args:
            namespace: Logical grouping such as ``temporal_query``.
            payload: All data that must match exactly, including query text.

        Returns:
            A namespaced key containing the payload digest.
        """
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        digest = hashlib.blake2b(blob.encode(), digest_size=16).hexdigest()
        return f"mindbridge:{namespace}:{digest}"

    @staticmethod
    def fingerprint(payload: dict[str, Any]) -> str:
        """Hash of everything that is NOT the query text.

        Semantic matching is allowed to be fuzzy about the question. It is
        never allowed to be fuzzy about the parameters that decide what a
        result even contains, so those are compared exactly.

        中文：查询文本可以模糊匹配，但决定结果内容的参数必须精确
        匹配。

        Args:
            payload: Non-query parameters such as limits, windows, and filters.

        Returns:
            A short digest used as a hard partition key.
        """
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.blake2b(blob.encode(), digest_size=8).hexdigest()

    @staticmethod
    def _index_key(namespace: str) -> str:
        """Return the Redis key for one namespace's semantic index.

        中文：返回某个命名空间的语义候选索引键。
        """
        # Deliberately under the namespace prefix so invalidate_namespace()
        # sweeps it away with the entries it describes.
        # 中文：索引沿用命名空间前缀，使 invalidate_namespace() 能同时清理它。
        return f"mindbridge:{namespace}:{_SEMANTIC_INDEX_SUFFIX}"

    # --- exact path -------------------------------------------------------

    async def get(self, key: str) -> dict[str, Any] | None:
        """Exact-key lookup: try the in-process LRU first, then Redis.

        中文：先查进程内 LRU，再查 Redis；Redis 命中会回填 LRU。

        Args:
            key: Exact cache key produced by ``QueryCache.key``.

        Returns:
            The cached value, or None on a miss.
        """
        local = self._lru.get(key)
        if local is not None:
            self.lru_hits += 1
            self.hits += 1
            return local
        if self._client is None:
            self.misses += 1
            return None
        raw = await self._client.get(key)
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        value = json.loads(raw)
        self._lru.set(key, value)
        return value

    async def set(self, key: str, value: dict[str, Any]) -> None:
        """Write value into both the LRU layer and Redis (if connected).

        中文：把值写入 LRU，并在已连接时同时写入 Redis。

        Args:
            key: Exact cache key.
            value: Result to cache.
        """
        self._lru.set(key, value)
        if self._client is None:
            return
        await self._client.set(
            key, json.dumps(value, ensure_ascii=False, default=str), ex=self._ttl
        )

    # --- semantic path ----------------------------------------------------

    async def index_semantic(
        self,
        namespace: str,
        key: str,
        query: str,
        fingerprint: str,
        embedding: list[float],
    ) -> None:
        """Register a freshly computed result as a semantic-match candidate.

        The supplied embedding is stored, not recomputed at lookup time. The
        semantic score is its dot product with the lookup embedding. Hashing
        and Ollama vectors are L2-normalised in this codebase; hosted providers
        are stored as returned, so their thresholds require separate validation.

        中文：保存刚计算的结果及其向量，查找时无需重新计算嵌入。本地 provider
        会在代码中归一化；托管 provider 使用接口原样返回的向量，阈值需单独验证。

        Args:
            namespace: Logical grouping for the entry.
            key: Exact-match cache key referenced by the candidate.
            query: Normalised query text retained for diagnostics.
            fingerprint: Digest of the exact-match parameters.
            embedding: Query embedding on the provider's scoring scale.
        """
        if self._client is None or not self.semantic_enabled:
            return
        index_key = self._index_key(namespace)
        entry = json.dumps(
            {
                "key": key,
                "q": query,
                "fp": fingerprint,
                "e": embedding,
                "exp": time.time() + self._ttl,
            },
            ensure_ascii=False,
        )
        pipeline = self._client.pipeline()
        # A list, not a set: insertion order is what lets the index be trimmed
        # to the most recent N without a scan.
        pipeline.lrem(index_key, 0, entry)
        pipeline.lpush(index_key, entry)
        pipeline.ltrim(index_key, 0, self._semantic_index_size - 1)
        pipeline.expire(index_key, self._ttl)
        await pipeline.execute()

    async def get_semantic(
        self, namespace: str, fingerprint: str, embedding: list[float]
    ) -> SemanticLookup:
        """Nearest cached query above threshold, with an ambiguity guard.

        Returns the reasons alongside the value: the eval needs to distinguish
        "nothing was close" from "two things were equally close", because only
        the second one indicates the embedder cannot support this feature.

        中文：返回值同时携带拒绝原因，使评测能区分“不够相近”和
        “候选歧义”。

        Args:
            namespace: Logical grouping to search.
            fingerprint: Exact partition for non-query parameters.
            embedding: Query embedding on the same scale as indexed vectors.

        Returns:
            A lookup result containing the value and decision trace.
        """
        if self._client is None or not self.semantic_enabled:
            return SemanticLookup(None, None, None, None, 0, "disabled")

        raw_entries = await self._client.lrange(self._index_key(namespace), 0, -1)
        now = time.time()
        scored: list[tuple[float, dict[str, Any]]] = []
        for raw in raw_entries:
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            # Hard partition: same question, different top_k, is a different
            # question as far as this cache is concerned.
            # 中文：参数不同（例如 top_k）时必须进入不同硬分区。
            if entry.get("fp") != fingerprint:
                continue
            if entry.get("exp", 0) <= now:
                continue
            vector = entry.get("e") or []
            if len(vector) != len(embedding):
                continue
            scored.append((_dot(vector, embedding), entry))

        if not scored:
            self.misses += 1
            return SemanticLookup(None, None, None, None, 0, "no_candidates")

        scored.sort(key=lambda item: item[0], reverse=True)
        best_similarity, best = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else None
        candidates = len(scored)

        def _refuse(outcome: str) -> SemanticLookup:
            self.misses += 1
            if outcome == "ambiguous":
                self.semantic_rejections += 1
            return SemanticLookup(
                None, best_similarity, runner_up, best.get("q"), candidates, outcome
            )

        if best_similarity < self.semantic_threshold:
            return _refuse("below_threshold")
        # Ambiguity guard. Two cached queries this close together means the
        # winner is arbitrary; serving it is a coin flip on which question gets
        # answered.
        # 中文：前两名过于接近时拒绝命中，避免随机回答其中一个问题。
        if runner_up is not None and (best_similarity - runner_up) < self.semantic_margin:
            return _refuse("ambiguous")

        raw_value = await self._client.get(best["key"])
        if raw_value is None:
            # Indexed but expired or evicted underneath us.
            # 中文：索引仍存在，但对应缓存值已经过期或被淘汰。
            self.misses += 1
            return SemanticLookup(
                None, best_similarity, runner_up, best.get("q"), candidates, "no_candidates"
            )

        self.hits += 1
        self.semantic_hits += 1
        return SemanticLookup(
            json.loads(raw_value),
            best_similarity,
            runner_up,
            best.get("q"),
            candidates,
            "served",
        )

    # --- invalidation -----------------------------------------------------

    async def invalidate_namespace(self, namespace: str) -> int:
        """Drop cached reads after a write, so a new preference is visible.

        This also removes the semantic index, which lives under the same
        prefix — otherwise a stale neighbour could keep answering for a
        preference that has since changed.

        中文：同时删除语义索引，避免旧邻居继续回答已变化的偏好。

        Args:
            namespace: Cache namespace to sweep.

        Returns:
            Number of Redis keys removed.
        """
        self._lru.clear()
        if self._client is None:
            return 0
        removed = 0
        async for key in self._client.scan_iter(f"mindbridge:{namespace}:*"):
            removed += await self._client.delete(key)
        return removed

    def stats(self) -> dict[str, Any]:
        """Snapshot of hit/miss counters by layer, for /healthz and evals.

        中文：按层级汇总命中与拒绝计数，供 /healthz 和评测使用。

        Returns:
            Counter and rate values for each cache layer.
        """
        looked_up = self.hits + self.misses
        return {
            "lookups": looked_up,
            "hits": self.hits,
            "lru_hits": self.lru_hits,
            "exact_redis_hits": self.hits - self.lru_hits - self.semantic_hits,
            "semantic_hits": self.semantic_hits,
            "semantic_rejections": self.semantic_rejections,
            "misses": self.misses,
            "hit_rate": (self.hits / looked_up * 100.0) if looked_up else None,
            "semantic_enabled": self.semantic_enabled,
            "semantic_threshold": self.semantic_threshold,
            "semantic_margin": self.semantic_margin,
        }

    def reset_stats(self) -> None:
        """Zero every hit/miss counter.

        中文：清零所有命中、拒绝和未命中计数器。
        """
        self.lru_hits = self.hits = self.semantic_hits = 0
        self.semantic_rejections = self.misses = 0

    async def close(self) -> None:
        """Clear the LRU and close the Redis connection, if any.

        中文：清空 LRU，并在存在连接时关闭 Redis 客户端。
        """
        self._lru.clear()
        if self._client is not None:
            await self._client.aclose()
