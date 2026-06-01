"""Optional standalone MCP server exposing Artoo's tools to external clients.

No longer used by the orchestrator — the boss now invokes these tools
in-process via agent_loop's tool dispatch. This module exists so that
external MCP clients (Claude.ai's qdrant-memory connector, artoo-web,
other agents) can call the same tool surface over stdio.

Run with: python -m artoo.mcp_server

Tools exposed (identical implementations to the in-process boss path):
  - search_memory: query qdrant + obsidian link graph
  - save_memory: persist a new memory
  - delete_memory: soft-archive (default) or hard-delete a memory
  - restore_memory: reverse a soft-archive
  - remind: schedule a one-shot reminder to Telegram
  - spawn_worker: dispatch to a specialist worker
"""
import asyncio
import json
from typing import Any

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions

from . import memory, reminders, workers

server: Server = Server("artoo")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    worker_names = workers.names()
    worker_catalog = workers.catalog()
    return [
        types.Tool(
            name="search_memory",
            description=(
                "Search Artoo's memory (Qdrant + obsidian link graph) for "
                "relevant past conversations, decisions, and notes about the "
                "user, their projects, and homelab. Use this whenever a user "
                "question references things they've told you before, or you "
                "need context you don't have from this conversation alone."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {
                        "type": "integer",
                        "default": 5,
                        "description": "Max primary results",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="save_memory",
            description=(
                "Save a new memory to Artoo's knowledge base (qdrant + obsidian). "
                "Use when the user shares a fact, preference, decision, plan, or "
                "important context that's worth recalling in future conversations.\n\n"
                "RECOMMENDED FLOW: call search_memory FIRST to find related existing "
                "memories — their UUIDs go into linked_uuids so the graph stays "
                "connected. Choose a short title (3-7 words) and 2-5 tags."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The memory content (verbatim or paraphrased)",
                    },
                    "title": {
                        "type": "string",
                        "description": "Short title (3-7 words)",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "2-5 tags categorizing this memory",
                    },
                    "linked_uuids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "UUIDs of related memories (from search_memory). Optional but recommended.",
                    },
                },
                "required": ["text"],
            },
        ),
        types.Tool(
            name="delete_memory",
            description=(
                "Delete a memory by UUID. Defaults to soft-archive: the "
                "memory becomes invisible to search but the content is "
                "preserved and can be brought back via restore_memory. "
                "Use this when a fact is stale, wrong, or duplicated by a "
                "newer/better memory.\n\n"
                "Set hard=true for irreversible deletion. Use hard only "
                "when you're sure (private data the user explicitly asked "
                "to forget, test residue, etc.) — soft-archive is almost "
                "always the right choice for stale/wrong content."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "uuid": {
                        "type": "string",
                        "description": "UUID of the memory to delete (from search_memory results)",
                    },
                    "hard": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "If true, permanently deletes the Qdrant point, "
                            "the obsidian page, the link-node, and scrubs "
                            "this UUID from all backlinks. Default false "
                            "(soft-archive)."
                        ),
                    },
                },
                "required": ["uuid"],
            },
        ),
        types.Tool(
            name="restore_memory",
            description=(
                "Restore a previously soft-archived memory. Sets status "
                "back to 'active' so it shows up in normal search again. "
                "Cannot restore hard-deleted memories — those are gone."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "uuid": {
                        "type": "string",
                        "description": "UUID of the archived memory to restore",
                    },
                },
                "required": ["uuid"],
            },
        ),
        types.Tool(
            name="remind",
            description=(
                "Schedule a one-shot reminder. Parses a human time delta "
                "(e.g. '30 minutes', '1h', '2h 30m', '45m', '1 day') and "
                "sends `text` to the operator's Telegram at that time. Self-cleans "
                "after firing — no leftover state.\n\n"
                "Use this whenever the operator asks to be reminded of something at "
                "a future time. Return the human-friendly confirmation to him."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "delta": {
                        "type": "string",
                        "description": "Time delta from now, e.g. '30 minutes', '2h', '1d 4h 15m'",
                    },
                    "text": {
                        "type": "string",
                        "description": "The reminder message to send when it fires",
                    },
                },
                "required": ["delta", "text"],
            },
        ),
        types.Tool(
            name="spawn_worker",
            description=(
                "Dispatch a focused task to a specialist worker. Workers "
                "have their own model + system prompt + scoped capabilities. "
                "Returns the worker's final text result.\n\n"
                f"Available workers:\n{worker_catalog}"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": worker_names,
                        "description": "Worker name",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Task description for the worker",
                    },
                },
                "required": ["name", "prompt"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    if name == "search_memory":
        result = await asyncio.to_thread(
            memory.search_memory,
            arguments["query"],
            arguments.get("limit", 5),
        )
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    if name == "save_memory":
        result = await asyncio.to_thread(
            memory.save_memory,
            arguments["text"],
            title=arguments.get("title"),
            tags=arguments.get("tags"),
            linked_uuids=arguments.get("linked_uuids"),
        )
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    if name == "delete_memory":
        try:
            result = await asyncio.to_thread(
                memory.delete_memory,
                arguments["uuid"],
                hard=bool(arguments.get("hard", False)),
            )
        except ValueError as e:
            return [types.TextContent(type="text", text=f"error: {e}")]
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    if name == "restore_memory":
        try:
            result = await asyncio.to_thread(memory.restore_memory, arguments["uuid"])
        except ValueError as e:
            return [types.TextContent(type="text", text=f"error: {e}")]
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    if name == "remind":
        try:
            delta_s = reminders.parse_delta(arguments["delta"])
        except ValueError as e:
            return [types.TextContent(type="text", text=f"error: {e}")]
        rid = await asyncio.to_thread(
            reminders.schedule, delta_s, arguments["text"]
        )
        # Friendly delta string back so the boss can echo "in 30 min" cleanly.
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "ok": True,
                "uuid": rid,
                "fires_in_seconds": delta_s,
                "text": arguments["text"],
            }),
        )]

    if name == "spawn_worker":
        result = await asyncio.to_thread(
            workers.run,
            arguments["name"],
            arguments["prompt"],
        )
        return [types.TextContent(type="text", text=result)]

    return [types.TextContent(type="text", text=f"unknown tool: {name}")]


async def amain():
    async with mcp.server.stdio.stdio_server() as (read, write):
        await server.run(
            read,
            write,
            InitializationOptions(
                server_name="artoo",
                server_version="0.0.1",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def main():
    asyncio.run(amain())


if __name__ == "__main__":
    main()
