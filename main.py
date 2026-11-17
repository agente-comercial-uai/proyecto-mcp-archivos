"""Punto de entrada del agente MCP de archivos.

Uso:
    python main.py archivos "pregunta" [--saltos] [--herramientas] [--diagrama]

Ejemplos:
    python main.py archivos "qué archivos hay en la carpeta"
    python main.py archivos "lee informe_enero.txt"
    python main.py archivos "busca la palabra consola en todos"
    python main.py archivos --herramientas
    python main.py archivos --diagrama
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()


def _aplanar(contenido) -> str:
    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, list):
        partes = []
        for b in contenido:
            if isinstance(b, dict) and b.get("type") == "text":
                partes.append(b.get("text", ""))
        return "".join(partes)
    return ""


def _texto_final(mensajes) -> str:
    for m in reversed(mensajes):
        t = _aplanar(getattr(m, "content", "") or "")
        if t.strip():
            return t
    return ""


def _mostrar_saltos(mensajes) -> int:
    saltos = 0
    for m in mensajes:
        tipo = getattr(m, "type", "")
        if tipo == "human":
            continue
        saltos += 1
        tool_calls = getattr(m, "tool_calls", None)
        etiqueta = f"[{tipo}]"
        if tool_calls:
            for tc in tool_calls:
                nombre = tc.get("name", "?")
                args = tc.get("args", {})
                print(f"{etiqueta} llamó a `{nombre}` con {args}")
        elif tipo == "tool":
            print(f"{etiqueta} devolvió:\n{_aplanar(getattr(m, 'content', ''))}")
        else:
            print(f"{etiqueta} {_aplanar(getattr(m, 'content', ''))}")
    return saltos


def _cmd_herramientas() -> None:
    from src.archivos.mcp_client import AgenteMCP

    async def run():
        a = AgenteMCP()
        await a.iniciar()
        hs = a.listar_herramientas()
        print(f"El servidor entregó {len(hs)} herramientas:\n")
        for nombre, desc in hs:
            print(f"- {nombre}: {desc}")

    asyncio.run(run())


def _cmd_chat(mensaje: str, saltos: bool) -> None:
    from langchain_core.messages import HumanMessage

    from src.archivos.mcp_client import AgenteMCP

    async def run():
        a = AgenteMCP()
        await a.iniciar()
        estado = await a.agente.ainvoke({"messages": [HumanMessage(content=mensaje)]})
        mensajes = estado.get("messages", [])
        if saltos:
            total = _mostrar_saltos(mensajes)
            print(f"\n-- total de pasos: {total} --\n")
        print(_texto_final(mensajes))

    asyncio.run(run())


def _cmd_diagrama() -> None:
    from src.archivos.mcp_client import AgenteMCP

    async def run():
        a = AgenteMCP()
        await a.iniciar()
        print(a.agente.get_graph().draw_mermaid())

    asyncio.run(run())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Agente MCP de archivos.")
    p.add_argument("agente", help="Nombre del agente (archivos)")
    p.add_argument("mensaje", nargs="?", help="Mensaje para el agente")
    p.add_argument("--saltos", action="store_true", help="Muestra qué herramienta llamó y qué devolvió.")
    p.add_argument("--herramientas", action="store_true", help="Lista las herramientas del servidor.")
    p.add_argument("--diagrama", action="store_true", help="Imprime el grafo Mermaid.")
    args = p.parse_args(argv)

    if args.herramientas:
        _cmd_herramientas()
        return 0

    if args.diagrama:
        _cmd_diagrama()
        return 0

    if not args.mensaje:
        print('Debes indicar un mensaje, por ejemplo: python main.py archivos "lista los archivos"')
        return 2

    _cmd_chat(args.mensaje, args.saltos)
    return 0


if __name__ == "__main__":
    sys.exit(main())
