"""
12-factor config with pydantic-settings.

Factor III: store config in the environment. Every variable here
can be set via env var (e.g. APP_DEBUG=true) or .env file.
No config value should ever be hardcoded in application code.
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── App ──────────────────────────────────────────────────────────────────
    app_name: str = "ai-engineer-handbook-01"
    app_version: str = "0.1.0"
    debug: bool = False
    workers: int = 1

    # ── Security ─────────────────────────────────────────────────────────────
    # In production: generate with `openssl rand -hex 32`
    secret_key: str = Field(default="dev-secret-change-in-prod", min_length=32)
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    stripe_webhook_secret: str = "whsec_test"

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./dev.db"

    # ── HTTP clients ─────────────────────────────────────────────────────────
    http_connect_timeout: float = 2.0
    http_read_timeout: float = 10.0
    http_max_connections: int = 100
    http_max_keepalive: int = 20

    # ── Retry / circuit breaker ───────────────────────────────────────────────
    retry_max_attempts: int = 3
    retry_base_delay: float = 0.5
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: float = 30.0

    # ── External services ────────────────────────────────────────────────────
    anthropic_api_key: str = Field(default="", description="Set via ANTHROPIC_API_KEY env var")
    llm_base_url: str = "https://api.anthropic.com"

    # ── CORS ─────────────────────────────────────────────────────────────────
    cors_origins: list[str] = ["http://localhost:3000"]

    @field_validator("secret_key")
    @classmethod
    def warn_default_secret(cls, v: str) -> str:
        if v == "dev-secret-change-in-prod":
            import warnings
            warnings.warn("Using default secret_key — override via SECRET_KEY env var", stacklevel=2)
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Nested env vars: HTTP_CONNECT_TIMEOUT=2.0
        # List env vars: CORS_ORIGINS='["https://app.com"]'
    )


settings = Settings()
