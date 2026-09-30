"""FastAPI transport over MemoryService."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request, status
from pydantic import BaseModel

from .agent_routes import router as agent_router
from .agent_runtime import MemoryAgentRuntime
from .local_agent import LocalMemoryAgent, OllamaChatClient
from .models import (
    CardScope,
    DailyReview,
    MemoryWithDecay,
    MemoryNamespace,
    MemoryMutationRequest,
    MemoryMutationResult,
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
)
from .service import MemoryService
from .settings import get_settings
from .run_ledger import AgentRunLedger
from .retrieval_eval_routes import router as retrieval_eval_router
from .night_shift_routes import router as night_shift_router

logging.basicConfig(level=logging.INFO)


class TurnWindow(BaseModel):
    """Turns in a range, plus how many exist beyond the returned page."""

    # 中文：时间范围内的原始记录页，并说明范围内的总记录数。

    turns: list[Turn]
    total: int
    returned: int


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start one shared MemoryService and close it with the FastAPI app.

    中文：随 FastAPI 应用启动共享的 MemoryService，并在应用停止时关闭它。

    Args:
        app: The application that stores the service on its state object.
    """
    service = await MemoryService.start(get_settings())
    app.state.service = service
    settings = get_settings()
    ledger = AgentRunLedger(service._pool)
    runtime = MemoryAgentRuntime(service, ledger)
    app.state.agent_runtime = runtime
    app.state.local_agent = LocalMemoryAgent(
        runtime,
        ledger,
        OllamaChatClient(
            settings.ollama_url,
            settings.agent_model,
            settings.agent_timeout_seconds,
        ),
    )
    try:
        yield
    finally:
        await service.close()


app = FastAPI(
    title="MindBridge memory API",
    version="0.1.0",
    summary="Three-tier long-term memory for LLM clients.",
    lifespan=lifespan,
)
app.include_router(agent_router)
app.include_router(retrieval_eval_router)
app.include_router(night_shift_router)


def get_service(request: Request) -> MemoryService:
    """Read the application-scoped MemoryService for a request.

    中文：从请求所属应用中取得共享的 MemoryService。

    Args:
        request: Incoming request whose application state holds the service.

    Returns:
        The initialised memory service.

    Raises:
        HTTPException: If a request arrives before the lifespan initialises the
            service.
    """
    service: MemoryService | None = getattr(request.app.state, "service", None)
    if service is None:  # pragma: no cover - only if lifespan was skipped
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="memory service is not initialised",
        )
    return service


ServiceDep = Annotated[MemoryService, Depends(get_service)]
SessionId = Annotated[str, Path(min_length=1, max_length=128)]


@app.get("/healthz", tags=["ops"])
async def healthz(service: ServiceDep) -> dict[str, object]:
    # Return runtime health without exposing the service implementation.
    # 中文：返回运行时健康状态，不暴露服务实现细节。
    result = await service.health()
    if result["status"] != "ok":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=result,
        )
    return result


# --- T1 -------------------------------------------------------------------


@app.post(
    "/sessions/{session_id}/turns",
    tags=["T1 session buffer"],
    status_code=status.HTTP_201_CREATED,
)
async def append_turn(
    session_id: SessionId, turn: TurnCreate, service: ServiceDep
) -> Turn:
    # Append one raw T1 turn through the service shared by every transport.
    # 中文：通过所有传输层共用的服务追加一条原始 T1 记录。
    return await service.add_turn(session_id, turn)


@app.get("/sessions/{session_id}/buffer", tags=["T1 session buffer"])
async def read_buffer(session_id: SessionId, service: ServiceDep) -> SessionBuffer:
    # Read the bounded live T1 window for one session.
    # 中文：读取某个会话受窗口限制的实时 T1 内容。
    return await service.read_buffer(session_id)


# --- T2 -------------------------------------------------------------------


@app.post(
    "/summaries",
    tags=["T2 rolling summary"],
    status_code=status.HTTP_201_CREATED,
)
async def write_summary(card: SummaryCardCreate, service: ServiceDep) -> SummaryCard:
    # Create or replace a structured T2 summary card.
    # 中文：创建或替换结构化的 T2 摘要卡。
    return await service.write_summary(card)


@app.get("/summaries", tags=["T2 rolling summary"])
async def list_summaries(
    service: ServiceDep,
    session_id: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=365)] = 30,
    scope: Annotated[CardScope, Query()] = "day",
) -> list[SummaryCard]:
    """Day cards by default.

    Session cards live in the same table, so the scope is explicit: without it
    the diary's day list would silently fill with hundreds of session rows the
    moment per-session cards were written.
    """
    # 中文：默认读取日卡；范围必须明确，避免会话卡填满日记列表。
    return await service.list_summaries(session_id, limit, scope)


# --- T3 -------------------------------------------------------------------


@app.post("/memories", tags=["T3 vector memory"])
async def upsert_preference(
    request: UpsertPreferenceRequest, service: ServiceDep
) -> UpsertPreferenceResult:
    """Same code path as the MCP `upsert_preference` tool."""
    # 中文：与 MCP 的 ``upsert_preference`` 工具走同一条业务路径。
    return await service.upsert_preference(request)


@app.get("/memories", tags=["T3 vector memory"])
async def list_memories(
    service: ServiceDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    include_superseded: Annotated[bool, Query()] = True,
    namespace: Annotated[MemoryNamespace | None, Query()] = None,
) -> list[MemoryWithDecay]:
    """Newest-first listing for the diary timeline.

    Deliberately not a search: no query means no cosine term, so each row
    carries only its decay weight. This does not bump access_count — drawing a
    timeline is not the model recalling something.
    """
    # 中文：日记时间线按最新优先读取，展示不会增加记忆访问次数。
    namespaces = [namespace] if namespace is not None else None
    return await service.list_memories(limit, include_superseded, namespaces)


@app.get("/memories/{memory_id}", tags=["T3 vector memory"])
async def get_memory(
    memory_id: Annotated[int, Path(ge=1)],
    service: ServiceDep,
) -> MemoryWithDecay:
    # Fetch one T3 memory by its stable audit id.
    # 中文：根据稳定的审计 ID 读取一条 T3 记忆。
    try:
        return await service.get_memory(memory_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@app.patch("/memories/{memory_id}", tags=["T3 vector memory"])
async def mutate_memory(
    memory_id: Annotated[int, Path(ge=1)],
    request: MemoryMutationRequest,
    service: ServiceDep,
) -> MemoryMutationResult:
    # Archive or replace one T3 memory without deleting history.
    # 中文：归档或替换一条 T3 记忆，同时保留历史记录。
    try:
        if request.action == "archive":
            return await service.archive_memory(memory_id)
        return await service.edit_memory(memory_id, request)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc


@app.get("/turns", tags=["T1 session buffer"])
async def list_turns(
    service: ServiceDep,
    start: Annotated[datetime, Query(description="inclusive, ISO 8601")],
    end: Annotated[datetime, Query(description="exclusive, ISO 8601")],
    limit: Annotated[int, Query(ge=1, le=200)] = 40,
) -> TurnWindow:
    """Raw turns in a time range — what the 'look underneath' panel shows.

    The range is passed as timestamps rather than a date so the caller owns the
    timezone; the server never has to guess which midnight was meant.
    """
    # 中文：读取半开时间范围内的原始记录；调用方负责选择时区。
    turns, total = await service.list_turns_between(start, end, limit)
    return TurnWindow(turns=turns, total=total, returned=len(turns))


@app.post("/memories/query", tags=["T3 vector memory"])
async def temporal_query(
    request: TemporalQueryRequest, service: ServiceDep
) -> TemporalQueryResult:
    """Same code path as the MCP `temporal_query` tool."""
    # 中文：与 MCP 的 ``temporal_query`` 工具走同一条业务路径。
    return await service.temporal_query(request)


@app.post("/patterns", tags=["Reflective patterns"])
async def propose_pattern(
    request: PatternCandidateCreate, service: ServiceDep
) -> PatternCandidate:
    """Create a reviewable inference outside T3."""
    # 中文：在 T3 之外创建可审核的推断候选项。
    return await service.propose_pattern(request)


@app.get("/patterns", tags=["Reflective patterns"])
async def list_patterns(
    service: ServiceDep,
    pattern_status: Annotated[PatternStatus | None, Query(alias="status")] = "pending",
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[PatternCandidate]:
    # List reviewable inferences, optionally filtered by lifecycle state.
    # 中文：列出可审核的反思型推断，并可按生命周期状态过滤。
    return await service.list_patterns(status=pattern_status, limit=limit)


@app.post("/patterns/{candidate_id}/resolve", tags=["Reflective patterns"])
async def resolve_pattern(
    candidate_id: Annotated[int, Path(ge=1)],
    request: PatternDecisionRequest,
    service: ServiceDep,
) -> PatternCandidate:
    # Apply an explicit user decision to an inference candidate.
    # 中文：将用户明确作出的决定应用到反思型推断候选项。
    try:
        return await service.resolve_pattern(candidate_id, request)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@app.get("/daily-review", tags=["Companion review"])
async def daily_review(
    service: ServiceDep,
    period: Annotated[str, Query()] = "latest",
) -> DailyReview:
    # Assemble the cross-tier review surface for one day.
    # 中文：组合某一天跨记忆层级的审核视图。
    return await service.daily_review(period)
