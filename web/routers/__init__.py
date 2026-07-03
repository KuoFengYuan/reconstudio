"""All API routers, collected for the app factory to include in one loop."""
from . import browse, create, doctor, jobs, measure, pages, viewer, viz

ROUTERS = [
    pages.router,
    browse.router,
    create.router,
    jobs.router,
    viz.router,
    viewer.router,
    measure.router,
    doctor.router,
]
