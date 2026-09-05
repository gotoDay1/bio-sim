import numpy as np

from bio_sim.core.growth import LogisticModel
from bio_sim.core.state import ReactorState
from bio_sim.core.simulation import BioreactorSimulator


def test_run():
    growth_model = LogisticModel(X_max=10.0, mu_max=0.5)
    sim = BioreactorSimulator(growth_model=growth_model)
    initial_state = ReactorState(t=0.0, V=1.0, X=0.1)

    t_span = (0.0, 24.0)
    t_eval = np.linspace(0.0, 24.0, 100)
    result = sim.run(initial_state=initial_state, t_span=t_span, t_eval=t_eval)

    assert "t" in result
    assert "X" in result

    assert len(result["t"]) == 100
    assert len(result["X"]) == 100

    assert result["X"][0] == 0.1

    assert np.all(np.diff(result["X"]) >= 0)
    assert np.all(result["X"] <= 10.0)
