"""Statistics engine for project 20 (stage S4).

Provider-agnostic: every function works on a `Tracks` object (per-frame player positions in metres, a team per
player, the ball position), whatever produced them: our own broadcast pipeline, Metrica Sports tracking, or a
synthetic test. Only NumPy is required for the core statistics; formation matching uses SciPy's Hungarian solver
and the pass network uses networkx.
"""

from .tracks import Tracks

__all__ = ["Tracks"]
