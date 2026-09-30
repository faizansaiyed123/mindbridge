"""MemoryService — the one place the tiers are orchestrated.

Both transports call this: the REST API in api/main.py and the MCP tools in
mcp_server/server.py. Neither owns any memory logic, so `upsert_preference`
over MCP and `POST /memories` over HTTP cannot drift apart.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Self

import httpx

import asyncpg

from .cache import QueryCache
from .db import apply_schema, create_pool
from .embeddings import Embedder, build_embedder
from .memory_candidates import (
    MemoryCandidate,
    MemoryCandidateCreate,
    MemoryCandidateDecisionRequest,
    MemoryCandidateStatus,
    MemoryCandidateStore,
)
from .memory import (
    PatternCandidateStore,
    RollingSummaryStore,
    SessionBufferStore,
    VectorMemoryStore,
    count_tokens,
)
from .models import (
    CardScope,
    DailyReview,
    MemoryHit,
    MemoryNamespace,
    MemoryWithDecay,
    PatternCandidate,
    PatternCandidateCreate,
    PatternDecisionRequest,
    PatternStatus,
    SessionBuffer,
    SummaryCard,
    SummaryCardCreate,
    TemporalQueryRequest,
    TemporalQueryResult,
    Turn,
    TurnCreate,
    UpsertPreferenceRequest,
    UpsertPreferenceResult,
    MemoryMutationRequest,
    MemoryMutationResult,
)
from .settings import Settings, get_settings

logger = logging.getLogger(__name__)

_QUERY_NAMESPACE = "temporal_query"

# Above the dedup threshold an incoming fact is the same fact. Between these two
# it is close enough to be about the same subject but different enough to be a
# change of mind — the case `supersedes_conflicting` handles.
_CONFLICT_THRESHOLD = 0.75


class MemoryService:
    """Coordinate T1, T2, T3, cache, and pattern stores for both transports.

    中文：为 REST API 与 MCP 协调 T1、T2、T3、缓存及模式候选存储。
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        embedder: Embedder,
        cache: QueryCache,
        settings: Settings,
    ) -> None:
        """Wire already-created dependencies into one memory service.

        中文：将已创建的依赖组装为一个统一的记忆服务。

        Args:
            pool: Open PostgreSQL connection pool.
            embedder: Embedding implementation used for T3 similarity work.
            cache: Query cache shared by temporal retrieval.
            settings: Validated runtime configuration.
        """
        self.settings = settings
        self.embedder = embedder
        self.cache = cache
        self._pool = pool
        self.turns = SessionBufferStore(pool, settings.session_buffer_window)
        self.summaries = RollingSummaryStore(pool)
        self.patterns = PatternCandidateStore(pool)
        self.candidates = MemoryCandidateStore(pool)
        self.vectors = VectorMemoryStore(
            pool,
            decay_rate_per_day=settings.decay_rate_per_day,
            dedup_threshold=settings.dedup_threshold,
            superseded_penalty=settings.superseded_penalty,
        )

    # --- lifecycle --------------------------------------------------------

    @classmethod
    async def start(cls, settings: Settings | None = None) -> Self:
        """Create all runtime dependencies and return a ready service.

        中文：创建运行时依赖并返回可立即使用的服务。

        Args:
            settings: Optional validated settings; reads configured defaults when
                omitted.

        Returns:
            A service with its database pool, embedder, and cache connected.
        """
        settings = settings or get_settings()
        if settings.embedding_provider == "hashing":
            logger.warning(
                "embedding provider is 'hashing': lexical overlap only, not "
                "semantic. Set MINDBRIDGE_EMBEDDING_PROVIDER=openai|gemini for "
                "real retrieval quality."
            )
        pool = await create_pool(settings)
        await apply_schema(pool, settings)
        embedder = build_embedder(settings)
        cache = await QueryCache.connect(settings)
        return cls(pool, embedder, cache, settings)

    async def close(self) -> None:
        """Close cache and database resources owned by this service.

        中文：关闭该服务持有的缓存、嵌入连接与数据库资源。
        """
        await self.cache.close()
        # The embedder now keeps its HTTP connection open across calls, so it
        # has to be closed with everything else or the loop shuts down with a
        # live transport attached.
        await self.embedder.aclose()
        await self._pool.close()

    async def health(self) -> dict[str, object]:
        """Report dependency health and safe runtime configuration.

        Each dependency is probed independently so one failure does not hide
        the state of the others.
        """
        checks: dict[str, str] = {}
        errors: dict[str, str] = {}

        try:
            await self._pool.fetchval("SELECT 1")
        except Exception as error:
            checks["postgres"] = "error"
            errors["postgres"] = f"PostgreSQL probe failed: {type(error).__name__}"
        else:
            checks["postgres"] = "ok"

        cache_health = await self.cache.health()
        checks["cache"] = cache_health["status"]
        if cache_health["status"] == "error":
            errors["cache"] = cache_health.get("detail", "Redis health check failed")

        try:
            await self.embedder.embed(["healthcheck"])
        except httpx.HTTPError as error:
            checks["embedder"] = "error"
            request = getattr(error, "request", None)
            endpoint = str(request.url) if request is not None else "configured endpoint"
            errors["embedder"] = (
                f"{self.embedder.name} {endpoint}: {type(error).__name__}"
            )
        except Exception as error:
            checks["embedder"] = "error"
            errors["embedder"] = f"{self.embedder.name}: {type(error).__name__}"
        else:
            checks["embedder"] = "ok"

        memories: int | None = None
        if checks["postgres"] == "ok":
            try:
                memories = await self.vectors.count()
            except Exception as error:
                checks["postgres"] = "error"
                errors["postgres"] = (
                    f"PostgreSQL probe failed: {type(error).__name__}"
                )

        return {
            "status": "ok" if not errors else "error",
            "postgres": checks["postgres"],
            "cache": checks["cache"],
            "embedder": self.embedder.name,
            "embedder_status": checks["embedder"],
            "checks": checks,
            "errors": errors,
            "memories": memories,
            "settings": self.settings.masked(),
        }

    # --- T1 ---------------------------------------------------------------

    async def add_turn(self, session_id: str, turn: TurnCreate) -> Turn:
        """Append one raw T1 turn to a session.

        中文：向一个会话追加一条原始 T1 记录。
        """
        return await self.turns.append(session_id, turn)

    async def read_buffer(self, session_id: str) -> SessionBuffer:
        """Read the bounded live T1 window for a session.

        中文：读取一个会话受窗口限制的实时 T1 内容。
        """
        return await self.turns.read(session_id)

    # --- T2 ---------------------------------------------------------------

    async def write_summary(self, card: SummaryCardCreate) -> SummaryCard:
        """Create or replace a structured T2 summary card.

        中文：创建或替换结构化 T2 摘要卡。
        """
        return await self.summaries.upsert(card)

    async def list_summaries(
        self,
        session_id: str | None = None,
        limit: int = 30,
        scope: CardScope = "day",
    ) -> list[SummaryCard]:
        """List T2 cards within an explicit scope.

        中文：在明确范围内列出 T2 摘要卡。
        """
        return await self.summaries.list_cards(session_id, limit, scope)

    async def get_summary(
        self, period: str, session_id: str | None = None
    ) -> SummaryCard | None:
        """Read one T2 card without exposing the store through a transport."""
        # 中文：读取一张 T2 卡，不让传输层直接依赖底层存储。
        return await self.summaries.get(period, session_id)

    # --- T3 ---------------------------------------------------------------

    async def upsert_preference(
        self, request: UpsertPreferenceRequest
    ) -> UpsertPreferenceResult:
        """Dedup-then-write.

        1. Embed the incoming fact.
        2. Find the nearest still-open record in the same namespace/category.
        3. At or above dedup_threshold: refresh it, insert nothing.
        4. Otherwise insert, and optionally close a conflicting neighbour.
        """
        # 中文：先嵌入并去重，再按阈值刷新、插入或替换 T3 偏好。
        [embedding] = await self.embedder.embed([request.content])
        match = await self.vectors.nearest_open(
            embedding, request.namespace, request.category
        )
        similarity = match.similarity if match else None

        if match is not None and similarity is not None:
            if similarity >= self.settings.dedup_threshold:
                record = await self.vectors.refresh(match.record.id)
                await self.cache.invalidate_namespace(_QUERY_NAMESPACE)
                return UpsertPreferenceResult(
                    action="refreshed",
                    record=record,
                    matched_id=match.record.id,
                    matched_similarity=similarity,
                    reason=(
                        f"cosine {similarity:.3f} >= threshold "
                        f"{self.settings.dedup_threshold:.2f}: same fact, "
                        "refreshed instead of duplicated"
                    ),
                )

        record = await self.vectors.insert(
            request.content,
            request.namespace,
            request.category,
            embedding,
            request.decay_factor,
            request.project,
        )

        action = "inserted"
        reason = (
            f"nearest open record cosine {similarity:.3f} < threshold "
            f"{self.settings.dedup_threshold:.2f}: new fact"
            if similarity is not None
            else "no existing record in this namespace/category: new fact"
        )
        if (
            request.supersedes_conflicting
            and match is not None
            and similarity is not None
            and similarity >= _CONFLICT_THRESHOLD
        ):
            await self.vectors.supersede(match.record.id, record.id)
            action = "superseded"
            reason = (
                f"cosine {similarity:.3f} is in the conflict band "
                f"[{_CONFLICT_THRESHOLD:.2f}, {self.settings.dedup_threshold:.2f}): "
                f"closed record {match.record.id} and replaced it"
            )

        await self.cache.invalidate_namespace(_QUERY_NAMESPACE)
        return UpsertPreferenceResult(
            action=action,  # type: ignore[arg-type]
            record=record,
            matched_id=match.record.id if match else None,
            matched_similarity=similarity,
            reason=reason,
        )

    async def list_memories(
        self,
        limit: int = 50,
        include_superseded: bool = True,
        namespaces: list[MemoryNamespace] | None = None,
    ) -> list[MemoryWithDecay]:
        """List newest T3 memories without semantic ranking.

        中文：按最新优先列出 T3 记忆，不进行语义排序。
        """
        return await self.vectors.list_recent(
            limit=limit,
            include_superseded=include_superseded,
            namespaces=namespaces,
        )

    async def get_memory(self, memory_id: int) -> MemoryWithDecay:
        """Read one T3 memory with its current decay weight.

        中文：读取一条 T3 记忆及其当前时间衰减权重。
        """
        return await self.vectors.get_with_decay(memory_id)

    async def archive_memory(self, memory_id: int) -> MemoryMutationResult:
        """Close one T3 memory without deleting its audit history.

        中文：关闭一条 T3 记忆，但不删除其审计历史。
        """
        record = await self.vectors.archive(memory_id)
        await self.cache.invalidate_namespace(_QUERY_NAMESPACE)
        return MemoryMutationResult(
            action="archive",
            target_id=record.id,
            replacement_id=record.id,
            memory=await self.vectors.get_with_decay(record.id),
            replacement_reason="manual archive requested",
        )

    async def edit_memory(
        self,
        memory_id: int,
        request: MemoryMutationRequest,
    ) -> MemoryMutationResult:
        """Replace one open T3 memory using confirmed edited wording.

        中文：使用确认后的编辑文本替换一条仍有效的 T3 记忆。
        """
        existing = await self.vectors.get(memory_id)
        if existing.valid_at is not None:
            raise ValueError(f"memory {memory_id} is already closed")

        [embedding] = await self.embedder.embed([request.content])
        replacement = await self.vectors.edit(
            memory_id,
            request.content,
            embedding,
            decay_factor=request.decay_factor,
        )
        await self.cache.invalidate_namespace(_QUERY_NAMESPACE)
        return MemoryMutationResult(
            action="edit",
            target_id=memory_id,
            replacement_id=replacement.id,
            memory=await self.vectors.get_with_decay(replacement.id),
            replacement_reason="manual edit created a replacement",
        )

    # --- reflective candidates ------------------------------------------

    async def propose_memory_candidate(
        self, draft: MemoryCandidateCreate
    ) -> MemoryCandidate:
        """Stage model-extracted memory outside T3 until explicit review."""
        return await self.candidates.create(draft)

    async def list_memory_candidates(
        self,
        *,
        status: MemoryCandidateStatus | None = "pending",
        limit: int = 50,
    ) -> list[MemoryCandidate]:
        return await self.candidates.list(status=status, limit=limit)

    async def resolve_memory_candidate(
        self,
        candidate_id: int,
        request: MemoryCandidateDecisionRequest,
    ) -> MemoryCandidate:
        """Promote confirmed wording into T3, or reject it without a T3 write."""
        candidate = await self.candidates.get(candidate_id)
        if candidate is None or candidate.status != "pending":
            raise KeyError(f"memory candidate {candidate_id} not found or resolved")
        if request.decision == "reject":
            return await self.candidates.resolve(
                candidate_id,
                status="rejected",
                content=candidate.content,
                resolution_note=request.resolution_note,
                confirmed_memory_id=None,
            )

        content = (request.confirmed_content or candidate.content).strip()
        outcome = await self.upsert_preference(
            UpsertPreferenceRequest(
                content=content,
                namespace=candidate.namespace,
                category=candidate.category,
                project=candidate.project,
                confirmed_by_user=candidate.namespace == "reflective",
            )
        )
        return await self.candidates.resolve(
            candidate_id,
            status="edited" if request.decision == "edit" else "confirmed",
            content=content,
            resolution_note=request.resolution_note,
            confirmed_memory_id=outcome.record.id,
        )

    async def propose_pattern(
        self, draft: PatternCandidateCreate
    ) -> PatternCandidate:
        """Keep a repeated-behaviour inference outside T3 until confirmation."""
        # 中文：重复行为推断在用户确认前只保存在 T3 之外的候选区。
        return await self.patterns.create(draft)

    async def list_patterns(
        self,
        *,
        status: PatternStatus | None = "pending",
        limit: int = 20,
    ) -> list[PatternCandidate]:
        """List reflective inference candidates by lifecycle status.

        中文：按生命周期状态列出反思型推断候选项。
        """
        return await self.patterns.list(status=status, limit=limit)

    async def resolve_pattern(
        self,
        candidate_id: int,
        request: PatternDecisionRequest,
    ) -> PatternCandidate:
        """Confirm/edit into reflective T3, or reject without writing memory."""
        # 中文：确认或编辑后写入反思型 T3；拒绝时不写入任何记忆。
        candidate = await self.patterns.get(candidate_id)
        if candidate is None or candidate.status != "pending":
            raise KeyError(f"pattern candidate {candidate_id} not found or resolved")

        if request.decision == "reject":
            return await self.patterns.resolve(
                candidate_id,
                status="rejected",
                description=candidate.description,
                resolution_note=request.resolution_note,
                confirmed_memory_id=None,
            )

        content = (request.confirmed_content or candidate.description).strip()
        outcome = await self.upsert_preference(
            UpsertPreferenceRequest(
                content=content,
                namespace="reflective",
                category="confirmed_pattern",
                confirmed_by_user=True,
            )
        )
        return await self.patterns.resolve(
            candidate_id,
            status="edited" if request.decision == "edit" else "confirmed",
            description=content,
            resolution_note=request.resolution_note,
            confirmed_memory_id=outcome.record.id,
        )

    async def daily_review(self, period: str = "latest") -> DailyReview:
        """Join one T2 card, today's T3 writes and pending reflective candidates."""
        # 中文：汇总一张 T2 卡、当天的两类 T3 写入与待处理的反思候选项。
        if period.strip().lower() == "latest":
            cards = await self.list_summaries(limit=1, scope="day")
            card = cards[0] if cards else None
            review_period = card.period if card is not None else date.today().isoformat()
        else:
            review_period = period.strip()
            card = await self.get_summary(review_period)

        operational = await self.list_memories(
            limit=200,
            include_superseded=False,
            namespaces=["operational"],
        )
        reflective = await self.list_memories(
            limit=200,
            include_superseded=False,
            namespaces=["reflective"],
        )
        return DailyReview(
            period=review_period,
            card=card,
            operational_memories=[
                memory
                for memory in operational
                if memory.created_at.date().isoformat() == review_period
            ],
            reflective_memories=[
                memory
                for memory in reflective
                if memory.created_at.date().isoformat() == review_period
            ],
            pending_patterns=await self.list_patterns(status="pending", limit=20),
        )

    async def list_turns_between(
        self, start: datetime, end: datetime, limit: int = 40
    ) -> tuple[list[Turn], int]:
        """Read a paged T1 time range and its unpaged total count.

        中文：读取分页后的 T1 时间范围记录及该范围内的总数。
        """
        turns = await self.turns.list_between(start, end, limit)
        total = await self.turns.count_between(start, end)
        return turns, total

    async def temporal_query(
        self,
        request: TemporalQueryRequest,
        trace: dict[str, object] | None = None,
    ) -> TemporalQueryResult:
        """Three-layer read: LRU -> exact key -> semantic neighbour -> Postgres.

        `trace`, if given, is filled with which layer answered and why. Nothing
        in the product path passes it; evals/eval_memory_engine.py does, because
        a semantic cache cannot be evaluated on its hit rate alone — the
        interesting number is how often it served the *wrong* question, and that
        needs the similarity and the query it matched against.
        """
        # 中文：按 LRU、精确缓存、语义邻居和 Postgres 的顺序查询；trace 仅供评估。
        # Everything except the query text. Semantic matching may be fuzzy about
        # the question; it must never be fuzzy about these, because they decide
        # what a result contains rather than what it is about.
        params = {
            "k": request.top_k,
            "w": request.time_window_days,
            "c": sorted(request.categories) if request.categories else None,
            "n": sorted(request.namespaces) if request.namespaces else None,
            "p": request.project,
            "s": request.include_superseded,
            "r": request.ranking_mode,
        }
        normalised_query = request.query_string.strip().lower()
        cache_key = QueryCache.key(_QUERY_NAMESPACE, {"q": normalised_query, **params})
        fingerprint = QueryCache.fingerprint(params)

        cached = await self.cache.get(cache_key)
        if cached is not None:
            if trace is not None:
                trace["layer"] = "exact"
            return TemporalQueryResult.model_validate({**cached, "cache_hit": True})

        # From here on the embedding has already been paid for. A semantic hit
        # therefore saves the vector search, not the embedding call — the
        # cost model in the eval says so explicitly rather than implying that a
        # semantic hit is as cheap as an exact one.
        [embedding] = await self.embedder.embed([request.query_string])

        lookup = await self.cache.get_semantic(
            _QUERY_NAMESPACE, fingerprint, embedding
        )
        if trace is not None:
            trace["semantic_outcome"] = lookup.outcome
            trace["semantic_similarity"] = lookup.best_similarity
            trace["semantic_runner_up"] = lookup.runner_up_similarity
            trace["semantic_matched_query"] = lookup.matched_query
            trace["semantic_candidates"] = lookup.candidates
        if lookup.value is not None:
            if trace is not None:
                trace["layer"] = "semantic"
            return TemporalQueryResult.model_validate(
                {**lookup.value, "cache_hit": True, "query": request.query_string}
            )

        hits = await self.vectors.search(
            embedding,
            top_k=request.top_k,
            time_window_days=request.time_window_days,
            categories=request.categories,
            namespaces=request.namespaces,
            include_superseded=request.include_superseded,
            project=request.project,
            ranking_mode=request.ranking_mode,
        )
        result = TemporalQueryResult(
            query=request.query_string,
            hits=hits,
            decay_rate_per_day=self.settings.decay_rate_per_day,
            cache_hit=False,
            context_block=format_context(hits),
        )
        payload = result.model_dump(mode="json")
        await self.cache.set(cache_key, payload)
        await self.cache.index_semantic(
            _QUERY_NAMESPACE, cache_key, normalised_query, fingerprint, embedding
        )
        if trace is not None:
            trace["layer"] = "miss"
        return result


def format_context(hits: list[MemoryHit]) -> str:
    """Render hits as the compact block a model is meant to receive.

    Every line carries provenance — id, date, score — so the model can cite what
    it used and a reader can tell a fresh preference from a decayed one.
    """
    # 中文：将命中结果渲染为模型可直接使用、且带溯源信息的紧凑上下文块。
    if not hits:
        return "No stored memory matched this query."
    lines = ["Known T3 memories (most relevant first):"]
    for hit in hits:
        state = (
            "open"
            if hit.valid_at is None
            else f"superseded {hit.valid_at.date().isoformat()}"
        )
        lines.append(
            f"- [{hit.id}] {hit.content} "
            f"(namespace={hit.namespace}, category={hit.category}, "
            f"learned={hit.created_at.date().isoformat()}, "
            f"{state}, cosine={hit.cosine_similarity:.3f}, score={hit.score:.3f})"
        )
    return "\n".join(lines)


def context_token_count(hits: list[MemoryHit]) -> int:
    """Count tokens in the prompt-ready representation of memory hits.

    中文：统计记忆命中结果渲染为提示词上下文后的 token 数量。
    """
    return count_tokens(format_context(hits))
