"""Request-side services: pure helpers the routers call (job→path resolution,
non-destructive edited-model derivation, form→params validation). No FastAPI
imports here, so these stay unit-testable and free of HTTP concerns.
"""
