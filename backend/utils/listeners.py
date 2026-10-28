"""Run the API on its plain listener plus an optional TLS listener.

The desktop app, the stdio MCP shim and local agents all talk plain HTTP to
127.0.0.1, so that listener never changes. When VOICEBOX_TLS_CERT and
VOICEBOX_TLS_KEY are set, a second HTTPS listener (VOICEBOX_TLS_HOST,
default 0.0.0.0; VOICEBOX_TLS_PORT, default the main port) serves remote
callers from the same process. Remote callers are still gated by
SecurityMiddleware, so this only adds transport encryption.
"""

import asyncio
import logging
import os
from dataclasses import dataclass

import uvicorn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TLSListener:
    host: str
    port: int
    certfile: str
    keyfile: str


def tls_listener_from_env(default_port: int) -> TLSListener | None:
    """Build the TLS listener config from the environment, or None if unset."""
    certfile = os.environ.get("VOICEBOX_TLS_CERT")
    keyfile = os.environ.get("VOICEBOX_TLS_KEY")
    if not certfile and not keyfile:
        return None
    if not certfile or not keyfile:
        raise ValueError("VOICEBOX_TLS_CERT and VOICEBOX_TLS_KEY must be set together")
    return TLSListener(
        host=os.environ.get("VOICEBOX_TLS_HOST", "0.0.0.0"),
        port=int(os.environ.get("VOICEBOX_TLS_PORT", str(default_port))),
        certfile=certfile,
        keyfile=keyfile,
    )


async def serve_with_tls(app, host: str, port: int, tls: TLSListener, log_level: str = "info") -> None:
    """Serve plain HTTP on host:port and HTTPS on the TLS listener.

    Only the plain server runs the app lifespan, so models and the database
    are initialised once. The TLS listener starts after that startup has
    finished, and whichever server stops first (signal, /shutdown, bind
    error) takes the other down with it.
    """
    plain = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level=log_level))
    secure = uvicorn.Server(
        uvicorn.Config(
            app,
            host=tls.host,
            port=tls.port,
            log_level=log_level,
            lifespan="off",
            ssl_certfile=tls.certfile,
            ssl_keyfile=tls.keyfile,
        )
    )

    plain_task = asyncio.create_task(plain.serve())
    while not plain.started:
        if plain_task.done():
            await plain_task
            return
        await asyncio.sleep(0.05)

    logger.info("Starting TLS listener on %s:%s", tls.host, tls.port)
    secure_task = asyncio.create_task(secure.serve())

    done, pending = await asyncio.wait({plain_task, secure_task}, return_when=asyncio.FIRST_COMPLETED)
    plain.should_exit = True
    secure.should_exit = True
    await asyncio.gather(*pending, return_exceptions=True)
    for task in done:
        task.result()


def run(app, host: str, port: int, log_level: str = "info") -> None:
    """Entry point: plain uvicorn.run, or both listeners when TLS is configured."""
    tls = tls_listener_from_env(port)
    if tls is None:
        uvicorn.run(app, host=host, port=port, log_level=log_level)
        return
    asyncio.run(serve_with_tls(app, host, port, tls, log_level))
