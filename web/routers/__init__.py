"""All API routers, collected for the app factory to include in one loop."""
from . import browse, create, doctor, jobs, pages, viz

ROUTERS = [
    pages.router,
    browse.router,
    create.router,
    jobs.router,
    viz.router,
    doctor.router,
]
