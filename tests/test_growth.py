from bio_sim.core.growth import LogisticModel


def test_logistic_model_init():
    model = LogisticModel(A=10, mu_max=0.5)
    assert model.A == 10.0
    assert model.mu_max == 0.5
