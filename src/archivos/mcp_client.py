"""Conexión y agente sobre el servidor MCP de archivos.

Usa `langchain-mcp-adapters`: arranca el servidor (@modelcontextprotocol/
server-filesystem vía npx), pide su lista de herramientas y se las entrega a
create_agent tal cual, sin código a medida por herramienta.
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

_SYSTEM_PROMPT = (
    "Eres un asistente que lee archivos de una carpeta compartida. "
    "Usa SIEMPRE las herramientas del servidor de archivos para listar, leer "
    "o buscar. Responde solo con lo que devuelvan esas herramientas."
)

# Ruta absoluta de la carpeta docs, relativa al proyecto.
_DOCS_DIR = str(Path(__file__).resolve().parents[2] / "docs")


def _mcp_config() -> dict:
    cfg = {
        "filesystem": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                _DOCS_DIR,
            ],
            "transport": "stdio",
        }
    }
    return cfg


def _crear_modelo():
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    modelo = os.getenv("MODELO", "gemini-3.5-flash-lite")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY no esta configurada en .env")
    return ChatGoogleGenerativeAI(model=modelo, google_api_key=api_key)


class AgenteMCP:
    """Envuelve el cliente MCP y el agente para reutilizarlos por comando."""

    def __init__(self):
        self.client = MultiServerMCPClient(_mcp_config())
        self.modelo = _crear_modelo()
        self.herramientas = None
        self.agente = None

    async def iniciar(self):
        # get_tools() arranca el servidor (npx) y trae las herramientas.
        self.herramientas = await self.client.get_tools()
        self.agente = create_react_agent(
            self.modelo,
            tools=self.herramientas,
            prompt=SystemMessage(content=_SYSTEM_PROMPT),
        )

    def listar_herramientas(self) -> list[tuple[str, str]]:
        """Devuelve (nombre, descripcion) de cada herramienta del servidor."""
        if self.herramientas is None:
            return []
        return [(t.name, t.description or "") for t in self.herramientas]
