from typing import Protocol

from bio_sim.core.growth import LogisticModel
from bio_sim.core.state import ReactorState

import numpy as np
from scipy.integrate import solve_ivp


class GrowthModel(Protocol):
    def rhs(self, state: ReactorState) -> float: ...


class BioreactorSimulator:
    def __init__(self, growth_model: LogisticModel) -> None:
        self.growth_model = growth_model

    def rhs(self, t: float, y: np.ndarray) -> list[float]:
        state = ReactorState.from_array(t=t, y=y)
        dXdt = self.growth_model.rhs(state=state)
        return [0.0, dXdt] if len(y) == 2 else [dXdt]

    def run(
        self,
        initial_state: ReactorState,
        t_span: tuple[float, float],
        t_eval: np.ndarray | None,
    ) -> dict[str, np.ndarray]:
        sol = solve_ivp(
            fun=self.rhs,
            t_span=t_span,
            y0=initial_state.to_array(),
            t_eval=t_eval,
            method="RK45",
        )
        res = {"t": sol.t}
        for idx, name in enumerate(["V", "X"] if len(sol.y) == 2 else ["X"]):
            res[name] = sol.y[idx]

        return res
