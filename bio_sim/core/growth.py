from bio_sim.core.state import ReactorState


class LogisticModel:
    def __init__(self, X_max: float, mu_max: float) -> None:
        self.X_max = X_max
        self.mu_max = mu_max

    def rhs(self, state: ReactorState) -> float:
        return self.mu_max * state.X * (1 - state.X / self.X_max)
