"""Every router module must be registered: an unlisted one 404s silently.

`/verify` (the acceptance check) shipped with its router and templates but was
missing from ROUTERS, so the page was unreachable until a user typed the URL
and got a 404.
"""
import importlib
import pkgutil

import web.routers as routers_pkg
from web.routers import ROUTERS


def test_every_router_module_is_registered():
    registered = {id(router) for router in ROUTERS}
    for info in pkgutil.iter_modules(routers_pkg.__path__):
        router = getattr(importlib.import_module(f"web.routers.{info.name}"), "router", None)
        if router is not None:
            assert id(router) in registered, f"web/routers/{info.name}.py is not in ROUTERS"


def test_workspace_pages_are_routed():
    paths = {route.path for router in ROUTERS for route in router.routes}
    assert {"/", "/verify", "/ui/verify", "/doctor"} <= paths
