"""Provedor DeepSeek com JSON Mode e validação Pydantic."""
from __future__ import annotations

import json
import time
from functools import lru_cache
from typing import Type, TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import settings
from errors import (
    ConfigurationError,
    LLMAuthenticationError,
    LLMBillingError,
    LLMRateLimitError,
    LLMServiceError,
    LLMStructuredOutputError,
)

T = TypeVar("T", bound=BaseModel)


class _TransientDeepSeekError(Exception):
    pass


class LLMProvider:
    def __init__(self, client: OpenAI | None = None, model: str | None = None):
        if settings.LLM_PROVIDER.lower() != "deepseek":
            raise ConfigurationError("LLM_PROVIDER deve ser 'deepseek' nesta versão.")
        if not settings.DEEPSEEK_API_KEY and client is None:
            raise ConfigurationError(
                "Configure DEEPSEEK_API_KEY no arquivo .env antes de gerar análises."
            )

        self.model = model or settings.LLM_MODEL
        self._client = client or OpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )

    def _request(self, messages: list[dict]):
        @retry(
            retry=retry_if_exception_type(_TransientDeepSeekError),
            stop=stop_after_attempt(settings.LLM_MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            reraise=True,
        )
        def call():
            try:
                return self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=settings.LLM_MAX_TOKENS,
                )
            except AuthenticationError as exc:
                raise LLMAuthenticationError(
                    "A chave da DeepSeek foi recusada. Gere uma nova chave e atualize o .env."
                ) from exc
            except RateLimitError as exc:
                raise LLMRateLimitError(
                    "A API da DeepSeek atingiu o limite de requisições. Aguarde e tente novamente."
                ) from exc
            except BadRequestError as exc:
                raise LLMServiceError(
                    "A DeepSeek recusou os parâmetros da solicitação. Verifique modelo e formato."
                ) from exc
            except APIStatusError as exc:
                if exc.status_code == 402:
                    raise LLMBillingError(
                        "O saldo da conta DeepSeek é insuficiente. Verifique o faturamento da API."
                    ) from exc
                if exc.status_code in {429}:
                    raise LLMRateLimitError(
                        "A API da DeepSeek atingiu o limite de requisições."
                    ) from exc
                if exc.status_code in {500, 502, 503, 504}:
                    raise _TransientDeepSeekError(str(exc)) from exc
                raise LLMServiceError(
                    f"A DeepSeek retornou um erro HTTP {exc.status_code}."
                ) from exc
            except (APIConnectionError, TimeoutError) as exc:
                raise _TransientDeepSeekError(str(exc)) from exc

        try:
            return call()
        except _TransientDeepSeekError as exc:
            raise LLMServiceError(
                "A DeepSeek está indisponível ou instável após novas tentativas."
            ) from exc

    def health_check(self) -> dict:
        """Valida autenticação e disponibilidade do modelo sem gerar conteúdo comercial."""
        try:
            models = self._client.models.list()
            available = [getattr(item, "id", "") for item in getattr(models, "data", [])]
            return {
                "ok": self.model in available or not available,
                "model": self.model,
                "available_models": available,
            }
        except AuthenticationError as exc:
            raise LLMAuthenticationError("A chave da DeepSeek foi recusada.") from exc
        except APIStatusError as exc:
            if exc.status_code == 402:
                raise LLMBillingError("Saldo insuficiente na conta DeepSeek.") from exc
            raise LLMServiceError(f"A DeepSeek retornou HTTP {exc.status_code}.") from exc
        except (APIConnectionError, TimeoutError) as exc:
            raise LLMServiceError("Não foi possível conectar à DeepSeek.") from exc

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
    ) -> tuple[T, dict]:
        schema = response_model.model_json_schema()
        schema_text = json.dumps(schema, ensure_ascii=False)

        base_system = (
            f"{system_prompt}\n\n"
            "Responda SOMENTE com um objeto JSON válido, sem markdown e sem texto adicional. "
            "O JSON deve respeitar integralmente este JSON Schema:\n"
            f"{schema_text}"
        )

        validation_feedback = ""
        started = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(1, settings.LLM_MAX_RETRIES + 1):
            messages = [
                {"role": "system", "content": base_system + validation_feedback},
                {"role": "user", "content": user_prompt},
            ]
            response = self._request(messages)
            content = response.choices[0].message.content or ""
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            if finish_reason == "length":
                last_error = ValueError("Resposta JSON truncada pelo limite de tokens.")
                validation_feedback = (
                    "\nA resposta anterior foi truncada. Gere um JSON mais conciso, preservando todos os campos obrigatórios."
                )
                continue
            if not content.strip():
                last_error = ValueError("A API retornou conteúdo vazio em JSON Mode.")
                validation_feedback = "\nA resposta anterior veio vazia. Gere apenas um objeto JSON válido e completo."
                continue

            try:
                parsed = response_model.model_validate_json(content)
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                usage = getattr(response, "usage", None)
                metadata = {
                    "modelo": f"deepseek/{self.model}",
                    "duracao_ms": elapsed_ms,
                    "custo_estimado_usd": None,
                    "input_tokens": getattr(usage, "prompt_tokens", None),
                    "output_tokens": getattr(usage, "completion_tokens", None),
                    "attempts": attempt,
                }
                return parsed, metadata
            except ValidationError as exc:
                last_error = exc
                validation_feedback = (
                    "\nA resposta anterior não respeitou o schema. Corrija estes erros e gere "
                    f"novamente apenas o JSON: {exc.errors(include_url=False)}"
                )
            except ValueError as exc:
                last_error = exc
                validation_feedback = (
                    "\nA resposta anterior não era JSON válido. Gere novamente apenas o objeto JSON."
                )

        raise LLMStructuredOutputError(
            "A DeepSeek respondeu, mas não foi possível validar a estrutura após "
            f"{settings.LLM_MAX_RETRIES} tentativas."
        ) from last_error


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    return LLMProvider()
