from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Production configuration with dual LLM provider support ($0 stack)."""
    
    # LLM Configuration
    # Defaults to local Ollama; can be overridden via environment variables for Google AI Studio
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "llama3.2:3b"
    llm_temperature: float = 0.0

    # Operational SQLite Database Paths
    business_db_path: str = "ops_data.sqlite"
    checkpoint_db_path: str = "agent_checkpoints.sqlite"

    # Governance & Business Policy
    auto_seed_demo_data: bool = True
    max_auto_refund_limit: float = 0.0  # By default, all refunds require HITL approval
    refund_window_days: int = 30

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def business_db_abs_path(self) -> Path:
        return Path(self.business_db_path).resolve()

    @property
    def checkpoint_db_abs_path(self) -> Path:
        return Path(self.checkpoint_db_path).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings singleton instance."""
    return Settings()
