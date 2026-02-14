"""FastAPI app entrypoint for the data centre simulator."""

from fastapi import FastAPI

from dc_sim.api.eval_routes import eval_router, set_eval_simulator
from dc_sim.api.routes import router, set_simulator
from dc_sim.config import SimConfig
from dc_sim.simulator import Simulator


def create_app(config: SimConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="DC Simulator", description="Simulated data centre for agent development")
    sim = Simulator(config)
    set_simulator(sim)
    set_eval_simulator(sim)
    app.include_router(router, tags=["simulator"])
    app.include_router(eval_router)
    return app


app = create_app()
