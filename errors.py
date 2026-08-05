"""Exceções de domínio apresentadas de forma segura na interface."""


class ProspectLLMError(Exception):
    """Erro base do sistema."""


class ConfigurationError(ProspectLLMError):
    pass


class DocumentSecurityError(ProspectLLMError):
    pass


class DocumentProcessingError(ProspectLLMError):
    pass


class LLMError(ProspectLLMError):
    pass


class LLMAuthenticationError(LLMError):
    pass


class LLMBillingError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    pass


class LLMServiceError(LLMError):
    pass


class LLMStructuredOutputError(LLMError):
    pass
