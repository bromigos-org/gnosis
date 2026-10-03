"""Process-scoped OpenAI-compatible clients for the LiteLLM endpoint.

Every LLM collaborator (router, recall filter, reranker, sufficiency, query
rewrite, graph-QA planner, bridge namer, fact extractor, community
summarizer) used to open ``async with AsyncOpenAI(...)`` per call: a fresh
httpx client, a fresh SSL context and a fresh TCP connection to LiteLLM on
every request, closed again when the call returned. One client per
``(base_url, api_key)`` keeps its connection pool across calls instead.

Clients are keyed by the running event loop too: an httpx pool belongs to
the loop it was opened on, so a process that runs more than one loop (the
test suite) never shares a pool across loops. ``close_shared_openai_clients``
closes the current loop's clients at shutdown.
"""

import asyncio
import weakref
from typing import Final

from openai import AsyncOpenAI

type _ClientKey = tuple[str, str]

_CLIENTS: Final[
    weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[_ClientKey, AsyncOpenAI]]
] = weakref.WeakKeyDictionary()


def shared_openai_client(*, api_key: str, base_url: str) -> AsyncOpenAI:
    """Return this loop's client for the endpoint, creating it on first use."""
    per_loop = _CLIENTS.setdefault(asyncio.get_running_loop(), {})
    key = (base_url, api_key)
    client = per_loop.get(key)
    if client is None:
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        per_loop[key] = client
    return client


async def close_shared_openai_clients() -> None:
    """Close and forget the current loop's clients (process shutdown)."""
    clients = _CLIENTS.pop(asyncio.get_running_loop(), {})
    for client in clients.values():
        await client.close()
