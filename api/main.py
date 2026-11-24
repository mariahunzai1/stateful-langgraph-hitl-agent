import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from core.config import Settings, get_settings
from core.database import get_db_connection, init_db, seed_demo_data
from core.graph import AgentService
from core.schemas import (
    AgentResumeRequest,
    AgentRunRequest,
    AgentTurnResponse,
    ApprovalDecision,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager initializing operational database and AgentService."""
    settings = get_settings()
    init_db(str(settings.business_db_abs_path))
    if settings.auto_seed_demo_data:
        seed_demo_data(str(settings.business_db_abs_path), force=False)

    app.state.settings = settings
    app.state.agent_service = AgentService(settings=settings)
    yield
    agent_svc = getattr(app.state, "agent_service", None)
    if agent_svc and hasattr(agent_svc, "_conn"):
        try:
            agent_svc._conn.close()
        except Exception:
            pass


app = FastAPI(
    title="Stateful LangGraph Operations Agent API",
    description="HITL-governed customer support and refund operations API with SQLite checkpoint persistence.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Dependency Injectors ---

def get_agent_service(request: Request) -> AgentService:
    """Dependency injector for AgentService instance."""
    service = getattr(request.app.state, "agent_service", None)
    if not service:
        # Fallback for dynamic contexts
        settings = getattr(request.app.state, "settings", get_settings())
        service = AgentService(settings=settings)
        request.app.state.agent_service = service
    return service


def get_settings_dep(request: Request) -> Settings:
    """Dependency injector for runtime Settings."""
    return getattr(request.app.state, "settings", get_settings())


# --- Endpoints ---

@app.get("/health")
def health(settings: Settings = Depends(get_settings_dep)) -> dict[str, str]:
    """Health check endpoint indicating model and provider configuration."""
    return {
        "status": "ok",
        "model": settings.llm_model,
        "base_url": settings.llm_base_url,
    }


@app.post("/agent/run", response_model=AgentTurnResponse)
def run_agent(
    req: AgentRunRequest,
    agent_service: AgentService = Depends(get_agent_service),
) -> AgentTurnResponse:
    """Execute an agent turn with user input."""
    try:
        return agent_service.run_turn(req.thread_id, req.message)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing agent turn: {str(exc)}",
        )


@app.post("/agent/resume", response_model=AgentTurnResponse)
def resume_agent(
    req: AgentResumeRequest,
    agent_service: AgentService = Depends(get_agent_service),
) -> AgentTurnResponse:
    """Resume a paused agent execution with an operator approval decision."""
    try:
        decision = req.decision or req.approval
        if not decision:
            raise ValueError("No decision provided.")
        return agent_service.resume_turn(req.thread_id, decision)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error resuming agent turn: {str(exc)}",
        )


@app.get("/threads/{thread_id}", response_model=AgentTurnResponse)
def get_thread_state(
    thread_id: str,
    agent_service: AgentService = Depends(get_agent_service),
) -> AgentTurnResponse:
    """Retrieve persisted thread state and active interrupts from SQLite checkpoint."""
    state = agent_service.get_thread_state(thread_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread '{thread_id}' not found or has no recorded state.",
        )
    return state


@app.get("/ops/snapshot")
def get_ops_snapshot(settings: Settings = Depends(get_settings_dep)) -> dict[str, list[dict[str, Any]]]:
    """Retrieve live SQLite records for real-time operator inspection."""
    db_path = str(settings.business_db_abs_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM customers ORDER BY id ASC;")
        customers = [dict(r) for r in cursor.fetchall()]

        cursor.execute("SELECT * FROM orders ORDER BY created_at DESC;")
        orders = [dict(r) for r in cursor.fetchall()]

        cursor.execute("SELECT * FROM refund_ledger ORDER BY created_at DESC;")
        refunds = [dict(r) for r in cursor.fetchall()]

        cursor.execute("SELECT * FROM audit_logs ORDER BY id DESC;")
        audit_raw = [dict(r) for r in cursor.fetchall()]

        audit_logs = []
        for entry in audit_raw:
            try:
                entry["payload"] = json.loads(entry["payload"])
            except Exception:
                pass
            audit_logs.append(entry)

        return {
            "customers": customers,
            "orders": orders,
            "refund_ledger": refunds,
            "audit_logs": audit_logs,
        }


@app.post("/ops/reset")
def reset_ops_data(settings: Settings = Depends(get_settings_dep)) -> dict[str, str]:
    """Reset and re-seed operational demo data for clean demonstrations."""
    db_path = str(settings.business_db_abs_path)
    seed_demo_data(db_path, force=True)
    return {
        "status": "reset_successful",
        "message": "Demo data reset successfully.",
    }
