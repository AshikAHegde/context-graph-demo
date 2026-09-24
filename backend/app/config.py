"""
Configuration management for the Context Graph application.
"""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Neo4jConfig:
    """Neo4j connection configuration."""

    uri: str
    username: str
    password: str
    database: Optional[str] = None

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        db = os.getenv("NEO4J_DATABASE")
        return cls(
            uri=os.getenv("NEO4J_URI", ""),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", ""),
            database=db if db else None,
        )


@dataclass
class GeminiConfig:
    """Google Gemini configuration for LLM agent and text embeddings."""

    api_key: str
    api_keys: list[str] = None  # type: ignore
    chat_model: str = "gemini-2.5-flash"
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768

    @classmethod
    def from_env(cls) -> "GeminiConfig":
        from .gemini_pool import parse_gemini_api_keys

        keys = parse_gemini_api_keys()
        primary_key = keys[0] if keys else os.getenv("GOOGLE_API_KEY", "")
        return cls(
            api_key=primary_key,
            api_keys=keys,
            chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash"),
            embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
            embedding_dimensions=int(os.getenv("GEMINI_EMBEDDING_DIMENSIONS", "768")),
        )


@dataclass
class AppConfig:
    """Main application configuration."""

    neo4j: Neo4jConfig
    gemini: GeminiConfig

    # FastRP embedding dimensions (structural)
    fastrp_dimensions: int = 128

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            neo4j=Neo4jConfig.from_env(),
            gemini=GeminiConfig.from_env(),
            fastrp_dimensions=int(os.getenv("FASTRP_DIMENSIONS", "128")),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            debug=os.getenv("DEBUG", "false").lower() == "true",
        )


# Global config instance
config = AppConfig.from_env()
