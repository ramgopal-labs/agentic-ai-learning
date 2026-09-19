import os
from typing import Literal

from app.core.config import is_usable_key, settings

Provider = Literal["openai", "groq", "gemini"]
Purpose = Literal["answer", "utility"]

# Two tiers per provider. Query rewriting and decomposition are short, structured
# transformations where the cheapest model is enough; the legal answer is the one
# call whose quality actually shows, so it gets the stronger model. Two of every
# three calls per question are utility calls, so the split is most of the cost.
ANSWER_MODELS: dict[str, str] = {
    "openai": "gpt-4.1-mini",
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.5-flash",
}

UTILITY_MODELS: dict[str, str] = {
    "openai": "gpt-4.1-nano",
    "groq": "llama-3.1-8b-instant",
    "gemini": "gemini-2.5-flash-lite",
}

# Kept as the answer-tier alias so existing callers and /health stay correct.
DEFAULT_MODELS = ANSWER_MODELS


def model_for(provider: str, purpose: Purpose = "answer") -> str:
    """
    The model to use, with an environment override per provider and purpose.

    e.g. OPENAI_ANSWER_MODEL=gpt-4.1 or GROQ_UTILITY_MODEL=llama-3.1-8b-instant
    """
    override = os.getenv(f"{provider.upper()}_{purpose.upper()}_MODEL")
    if override:
        return override

    table = ANSWER_MODELS if purpose == "answer" else UTILITY_MODELS
    return table[provider]


_KEY_ENV_NAMES: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


class MissingProviderKey(RuntimeError):
    """Raised when a provider is selected but its API key is absent or a placeholder."""


class LLMService:
    """
    One entry point for answer generation across OpenAI, Groq and Gemini.

    Clients are built lazily and cached, so selecting OpenAI never requires a
    Groq or Gemini key to be present.
    """

    def __init__(self, default_provider: Provider = "openai"):
        self.default_provider = default_provider
        self._clients: dict[str, object] = {}

    # --- key handling ------------------------------------------------------

    @staticmethod
    def _api_key(provider: str) -> str:
        key = {
            "openai": settings.openai_api_key,
            "groq": settings.groq_api_key,
            "gemini": settings.gemini_api_key,
        }.get(provider)

        if not is_usable_key(key):
            raise MissingProviderKey(
                f"{_KEY_ENV_NAMES[provider]} is missing or still a placeholder. "
                f"Add a real key to .env to use the '{provider}' provider."
            )
        return key  # type: ignore[return-value]

    def available_providers(self) -> list[str]:
        """Providers whose key is present, for surfacing in the API and UI."""
        return [
            provider
            for provider in ANSWER_MODELS
            if is_usable_key(
                {
                    "openai": settings.openai_api_key,
                    "groq": settings.groq_api_key,
                    "gemini": settings.gemini_api_key,
                }[provider]
            )
        ]

    # --- client construction ----------------------------------------------

    def _client(self, provider: str):
        if provider in self._clients:
            return self._clients[provider]

        api_key = self._api_key(provider)

        # Without an explicit timeout a hung provider holds the request open
        # until the caller gives up, tying up a worker the whole time.
        timeout = settings.llm_timeout_seconds
        retries = settings.llm_max_retries

        if provider == "openai":
            from openai import OpenAI

            client = OpenAI(api_key=api_key, timeout=timeout, max_retries=retries)
        elif provider == "groq":
            from groq import Groq

            client = Groq(api_key=api_key, timeout=timeout, max_retries=retries)
        elif provider == "gemini":
            from google import genai

            client = genai.Client(
                api_key=api_key,
                http_options={"timeout": int(timeout * 1000)},  # milliseconds
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        self._clients[provider] = client
        return client

    # --- generation --------------------------------------------------------

    def generate(
        self,
        prompt: str,
        provider: Provider | None = None,
        model: str | None = None,
        temperature: float = 0.1,
        purpose: Purpose = "answer",
    ) -> str:
        provider = provider or self.default_provider
        if provider not in ANSWER_MODELS:
            raise ValueError(f"Unknown provider: {provider}")

        model = model or model_for(provider, purpose)
        client = self._client(provider)

        if provider in ("openai", "groq"):
            # Groq's SDK mirrors the OpenAI chat-completions shape.
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response.choices[0].message.content or "No answer was generated."

        # Gemini takes the prompt string directly rather than a messages list.
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text or "No answer was generated."
