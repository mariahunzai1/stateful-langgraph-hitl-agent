# LangGraph Engineering Standards & Architectural Rules

## 1. Architectural Separation of Concerns
The codebase strictly adheres to layered domain boundaries. Cross-layer violations are disallowed.

- `core/`: Core engine domain containing:
  - `schemas.py`: Pydantic v2 schemas for agent state, tool inputs/outputs, approval requests, and audit models.
  - `config.py`: Centralized `Settings` loaded via `pydantic-settings`.
  - `database.py`: Operational SQLite manager with WAL mode, foreign keys, idempotent seed dataset, and atomic transactions.
  - `tools.py`: Tool definitions tagged programmatically with explicit `ToolRiskLevel` metadata.
  - `agent.py`: LangGraph state machine, interrupt definitions, SQLite checkpointer, and post-tool verification logic.
- `api/`: FastAPI REST layer:
  - Exposes `/run`, `/resume`, `/audit-logs`, and health endpoints.
  - Translates HTTP payloads into LangGraph thread invocations.
  - Enforces request validation using `core/schemas.py`.
- `ui/`: Streamlit HITL Operator Console:
  - **Zero Direct Import Rule**: The UI must **never** import from `core/` or open SQLite files.
  - Interacts exclusively with the FastAPI service via HTTP client (`httpx`).
  - Ensures clean process separation and eliminates SQLite database lock contention.
- `eval/`: Automated benchmark harness:
  - Measures agent decision accuracy, HITL gate adherence, policy adherence, and latency.
- `tests/`: 100% offline `pytest` suite:
  - Mocks all LLM interactions.
  - Uses isolated in-memory or temporary SQLite instances for fixtures.

## 2. Pydantic v2 Contract Enforcement
- All tool inputs must derive from `pydantic.BaseModel` with strict field constraints and descriptions.
- All tool outputs, API request bodies, and API responses must be strongly typed with Pydantic v2 models.
- Approval payloads and audit log records must validate against typed schemas (`PendingApproval`, `ApprovalDecision`, `AuditLogEntry`).

## 3. Dual LLM Compatibility ($0 Stack)
- Agents must interface with LLMs solely through `langchain-openai.ChatOpenAI`.
- Configure `base_url`, `api_key`, and `model` dynamically from `Settings`.
- Support:
  - Local Ollama: `base_url="http://localhost:11434/v1"`, `model="llama3.2:3b"`, `api_key="ollama"`.
  - Google AI Studio: `base_url="https://generativelanguage.googleapis.com/v1beta/openai/"`, `model="gemini-2.0-flash"`, `api_key="<GEMINI_API_KEY>"`.

## 4. Deterministic Risk-Tiering & Post-Tool Verification
- Tools must be tagged with `risk_level: ToolRiskLevel` (`READ_ONLY` vs `HIGH_RISK`).
- LangGraph conditional routers must inspect `tool.risk_level` programmatically—never rely on LLM self-tagging.
- High-risk operations (refunds, subscription updates) require Human-in-the-Loop review when exceeding configured thresholds.
- After any tool execution, deterministic post-tool verification must query the SQLite operations database to ensure state consistency before transitioning the graph.
