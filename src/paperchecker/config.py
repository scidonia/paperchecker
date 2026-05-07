"""Environment configuration — reads API keys and defaults from the environment."""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    """Configuration loaded from environment variables."""

    deepseek_api_key: str = field(
        default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY", "")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", "")
    )
    anthropic_api_key: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "")
    )

    semantic_scholar_api_key: str = field(
        default_factory=lambda: os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    )
    llm_max_tokens: int = 4096
    llm_timeout: int = 120

    manifests_dir: str = "_paperchecker"
    papers_dir: str = "_paperchecker/papers"

    @property
    def available_backends(self) -> list[str]:
        """Return names of LLM backends that have API keys configured."""
        backends = []
        if self.deepseek_api_key:
            backends.append("deepseek")
        if self.openai_api_key:
            backends.append("openai")
        if self.anthropic_api_key:
            backends.append("claude")
        return backends

    @property
    def preferred_backend(self) -> str | None:
        """Return the first available backend (DeepSeek preferred)."""
        backends = self.available_backends
        return backends[0] if backends else None
