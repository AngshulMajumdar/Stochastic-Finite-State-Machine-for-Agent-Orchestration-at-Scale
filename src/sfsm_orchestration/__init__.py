"""FSM/SFSM orchestration research implementation."""

from .core import (
    BenchmarkConfig,
    accuracy,
    fsm_select,
    generate_agent_graph,
    sfsm_log_posterior,
    sfsm_select,
)

__all__ = [
    "BenchmarkConfig", "accuracy", "fsm_select", "generate_agent_graph",
    "sfsm_log_posterior", "sfsm_select"
]
__version__ = "1.0.0"
