"""
LLM Service - OpenAI/Groq configuration for Pipecat
This module provides configuration and utilities for LLM services.
The actual LLM service is created directly in the pipeline using Pipecat's LLM services.
"""
from typing import Literal, Optional
from loguru import logger


class LLMConfig:
    """
    Configuration for LLM service (OpenAI or Groq).
    Used to configure the LLM parameters for the interview pipeline.
    """
    
    def __init__(
        self,
        provider: Literal["openai", "groq"] = "openai",
        api_key: str = "",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        fallback_provider: Optional[Literal["openai", "groq"]] = None,
        fallback_api_key: Optional[str] = None,
    ):
        self.provider = provider.lower()
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fallback_provider = fallback_provider
        self.fallback_api_key = fallback_api_key
        
        # Set default model based on provider
        if model is None:
            self.model = self._get_default_model(self.provider)
        else:
            self.model = model
        
        logger.info(
            f"LLMConfig initialized (provider: {provider}, model: {self.model}, "
            f"temperature: {temperature}, max_tokens: {max_tokens})"
        )
    
    def _get_default_model(self, provider: str) -> str:
        """Get default model for provider."""
        defaults = {
            "openai": "gpt-4",
            "groq": "llama-3.1-70b-versatile"
        }
        return defaults.get(provider, "gpt-4")
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for service initialization."""
        return {
            "api_key": self.api_key,
            "model": self.model,
        }


def create_llm_config(
    openai_api_key: str = "",
    groq_api_key: str = "",
    provider: Literal["openai", "groq"] = "openai",
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 500,
    enable_fallback: bool = True
) -> LLMConfig:
    """
    Factory function to create LLM configuration with optional fallback.
    
    Args:
        openai_api_key: OpenAI API key
        groq_api_key: Groq API key
        provider: Primary LLM provider ("openai" or "groq")
        model: Model name (uses default if not specified)
        temperature: Response temperature (0-1)
        max_tokens: Maximum tokens in response
        enable_fallback: Enable fallback to alternate provider
    
    Returns:
        LLMConfig instance
    """
    # Determine primary API key
    api_key = openai_api_key if provider == "openai" else groq_api_key
    
    # Configure fallback
    fallback_provider = None
    fallback_api_key = None
    if enable_fallback:
        if provider == "openai" and groq_api_key:
            fallback_provider = "groq"
            fallback_api_key = groq_api_key
        elif provider == "groq" and openai_api_key:
            fallback_provider = "openai"
            fallback_api_key = openai_api_key
    
    return LLMConfig(
        provider=provider,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        fallback_provider=fallback_provider,
        fallback_api_key=fallback_api_key
    )


def get_openai_llm_service(config: LLMConfig):
    """
    Get OpenAI LLM service instance for Pipecat pipeline.
    
    Args:
        config: LLM configuration
    
    Returns:
        Pipecat OpenAILLMService instance
    """
    try:
        from pipecat.services.openai import OpenAILLMService
        
        return OpenAILLMService(
            api_key=config.api_key,
            model=config.model,
        )
    except ImportError as e:
        logger.error(f"Failed to import OpenAI service: {e}")
        logger.error("Install with: pip install 'pipecat-ai[openai]'")
        raise


def get_groq_llm_service(config: LLMConfig):
    """
    Get Groq LLM service instance for Pipecat pipeline.
    
    Args:
        config: LLM configuration
    
    Returns:
        Pipecat GroqLLMService instance
    """
    try:
        from pipecat.services.groq import GroqLLMService
        
        return GroqLLMService(
            api_key=config.api_key,
            model=config.model,
        )
    except ImportError as e:
        logger.error(f"Failed to import Groq service: {e}")
        logger.error("Install with: pip install 'pipecat-ai[groq]' or pip install groq")
        raise


def get_llm_service(config: LLMConfig):
    """
    Get LLM service instance based on configuration.
    
    Args:
        config: LLM configuration
    
    Returns:
        Pipecat LLM service instance (OpenAI or Groq)
    """
    if config.provider == "groq":
        return get_groq_llm_service(config)
    else:
        return get_openai_llm_service(config)


# Supported models for each provider
SUPPORTED_MODELS = {
    "openai": [
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-3.5-turbo",
    ],
    "groq": [
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.2-90b-text-preview",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]
}


def list_supported_models(provider: str = "openai") -> list:
    """Get list of supported models for a provider."""
    return SUPPORTED_MODELS.get(provider.lower(), [])


def get_model_info(model: str) -> dict:
    """Get information about a specific model."""
    model_info = {
        "gpt-4": {
            "provider": "openai",
            "context_window": 8192,
            "description": "Most capable GPT-4 model"
        },
        "gpt-4o": {
            "provider": "openai",
            "context_window": 128000,
            "description": "Fast, multimodal GPT-4 model"
        },
        "llama-3.1-70b-versatile": {
            "provider": "groq",
            "context_window": 32768,
            "description": "Fast Llama 3.1 70B model via Groq"
        },
        "llama-3.1-8b-instant": {
            "provider": "groq",
            "context_window": 32768,
            "description": "Ultra-fast Llama 3.1 8B model via Groq"
        },
    }
    return model_info.get(model, {"description": "Unknown model"})