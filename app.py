"""Recon Studio — web front-end over the pure-Python pipeline modules.

Single-machine local tool. Binds to 127.0.0.1 by default. Run:
    ./run.sh            # or: uvicorn app:app --host 127.0.0.1 --port 8077

This module is just the **app factory**: build the FastAPI app, mount static
assets, start the job manager, and include the routers. The route handlers live
under `web/routers/`, shared render helpers in `web/shared.py`, and pure
request-side logic (job→path resolution, form validation) in `web/services/`.
The heavy work stays in the torch-free `pipeline/` package.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from jobs import manager
from web.routers import ROUTERS
from web.shared import BASE

app = FastAPI(title="Recon Studio")

(BASE / "static").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")

for _router in ROUTERS:
    app.include_router(_router)


@app.on_event("startup")
async def _startup() -> None:
    manager.start()
