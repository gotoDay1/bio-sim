from bio_sim.core.growth import LogisticModel
from bio_sim.core.state import ReactorState


def test_logistic_model_init():
    model = LogisticModel(X_max=10, mu_max=0.5)
    assert model.X_max == 10.0
    assert model.mu_max == 0.5


def test_logistic_model_rhs():
    model = LogisticModel(X_max=10, mu_max=0.5)
    state = ReactorState(t=1, V=1, X=1)
    assert model.rhs(state) == 9 / 20
