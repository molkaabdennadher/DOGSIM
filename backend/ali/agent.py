"""
agent.py — Ali: the shelter assignment ReAct agent.

Architecture
------------
Ali is a LangGraph ReAct agent backed by Groq llama-4-scout-17b.
She has two tool groups:
  QUERY_TOOLS   — read-only, always query the live DB
  ACTION_TOOLS  — write DB (assign dogs, release, add shelter, etc.)

Every tool that reads data opens a fresh SQLite connection so Ali always
sees the latest committed state — no stale cache.

Session state
-------------
Each WebSocket connection gets its own LangGraph thread_id so message
histories are isolated between browser sessions.

System prompt philosophy
------------------------
- Ali is direct, data-driven, and proactively surfaces insights.
- She ALWAYS queries the DB before answering data questions.
- She explains her reasoning in plain language, not raw JSON.
- For complex assignments she explains WHY each dog went where.
"""
from __future__ import annotations

import json
import os
import uuid
from typing import AsyncGenerator

from groq import Groq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from backend.ali.tools import ALL_TOOLS, set_groq_client
import backend.ali.db as db

GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

SYSTEM_PROMPT = """You are Ali, an expert animal welfare coordinator and AI assistant for a dog shelter management system.

## Core identity
You are a LIVE DATABASE ASSISTANT first. Your knowledge comes exclusively from real-time tool calls — never from memory, assumptions, or previous conversation turns. Every number you state must come from a tool call made in the current response.

## What you can always do (no file upload needed)
- Answer any question about the current state of shelters: capacity, free spots, budget, vet presence, expertise level.
- List all dogs currently assigned to any shelter.
- Show the waiting list (dogs not yet placed).
- Report system-wide statistics: total dogs, assigned, waiting, released, special conditions.
- Identify shelters at capacity or budget risk.
- Release a dog from a shelter and attempt to re-fill the vacancy from the waiting list.
- Add a new shelter to the system.

## Excel upload (optional feature)
When the user uploads an Excel file of incoming dogs, process the assignment batch:
- Sort by priority (Serious Injury +100, Pregnant +85, Skin Disease +60, Abnormal +50, Minor Injury +40, Not Vaccinated +15, Not Dewormed +10).
- Apply hard constraints: capacity, vet for serious injuries, budget, expertise for abnormal behaviour.
- After each batch, proactively flag capacity risks and waiting-list cases.

## Mandatory rules
- ALWAYS call a query tool before answering any question about data. Never state numbers from memory.
- If asked about shelter state, call get_all_shelters or get_shelter_details — even if you processed a file two messages ago, the DB may have changed.
- Compute effective_free_spots = free_spots − reserved_spots. Always show both numbers.
- When asked for statistics, call get_statistics. Do not compute from memory.
- Be concise. Use bullet points for lists. State exact numbers.
- If you cannot place a dog, explain which specific constraint failed and with which value.

You speak English and French."""


_groq_client: Groq | None    = None
_llm:         ChatGroq | None = None
_agent        = None
_memory       = MemorySaver()


def init_agent() -> None:
    """Initialise the Groq client, LLM, and LangGraph ReAct agent.

    Must be called at application startup (after load_dotenv has run).
    The API key is read fresh from the environment here — NOT at module-import
    time — so that .env values are guaranteed to be visible.
    """
    global _groq_client, _llm, _agent

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")

    _groq_client = Groq(api_key=api_key)
    set_groq_client(_groq_client)   # inject into action_tools

    _llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=api_key,
        temperature=0.2,
        max_tokens=2048,
    )

    _agent = create_react_agent(
        model=_llm,
        tools=ALL_TOOLS,
        checkpointer=_memory,
        prompt=SYSTEM_PROMPT,
    )

    # Ensure DB is initialised
    db.init_db()
    print(f"[Ali] Agent ready — model: {GROQ_MODEL}")


def get_thread_config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


async def chat(message: str, session_id: str) -> AsyncGenerator[str, None]:
    """
    Stream Ali's response token-by-token.

    Yields
    ------
    JSON strings of shape:
        {"type": "token",  "content": "..."}
        {"type": "tool",   "name": "...", "input": "...", "output": "..."}
        {"type": "done"}
        {"type": "error",  "content": "..."}
    """
    if _agent is None:
        yield json.dumps({"type": "error", "content": "Agent not initialised."})
        return

    config = get_thread_config(session_id)

    try:
        async for event in _agent.astream_events(
            {"messages": [HumanMessage(content=message)]},
            config=config,
            version="v2",
        ):
            kind = event.get("event", "")

            # ── Tool call started ──────────────────────────────────────────
            if kind == "on_tool_start":
                tool_name  = event.get("name", "tool")
                tool_input = event.get("data", {}).get("input", "")
                yield json.dumps({
                    "type":  "tool",
                    "phase": "start",
                    "name":  tool_name,
                    "input": str(tool_input)[:200],
                })

            # ── Tool call finished ─────────────────────────────────────────
            elif kind == "on_tool_end":
                tool_name   = event.get("name", "tool")
                tool_output = event.get("data", {}).get("output", "")
                yield json.dumps({
                    "type":   "tool",
                    "phase":  "end",
                    "name":   tool_name,
                    "output": str(tool_output)[:500],
                })

            # ── LLM token stream ───────────────────────────────────────────
            elif kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield json.dumps({"type": "token", "content": chunk.content})

        yield json.dumps({"type": "done"})

    except Exception as exc:
        yield json.dumps({"type": "error", "content": str(exc)})


def chat_sync(message: str, session_id: str) -> str:
    """Non-streaming variant — returns Ali's full response as a string.
    Used by the REST fallback endpoint.
    """
    if _agent is None:
        return "Agent not initialised."

    config = get_thread_config(session_id)
    result = _agent.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=config,
    )
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return "No response generated."
