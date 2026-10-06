import numpy as np
from probe_harness.transitions import get_standard_transitions
from probe_harness.train_and_measure import train_and_measure
from probe_harness.report import print_comparison_table
from pcn_core.model import PredictiveCodingNetwork

def test_get_standard_transitions():
    tA, tB, tC = get_standard_transitions()
    assert len(tA) > 0
    assert len(tB) > 0
    assert len(tC) > 0

def test_train_and_measure():
    # Use very small epochs for fast test
    res = train_and_measure(
        model_class=PredictiveCodingNetwork,
        model_kwargs={'layer_sizes': [7, 8, 2]},
        epochs=1,
        seed=42
    )
    assert len(res) == 5
    for r in res:
        assert isinstance(r, float)

def test_print_comparison_table():
    # Test does not raise exception
    results = {
        'Baseline': (0.1, 0.2, 0.1, 0.3, 0.2),
        'Exp1': (0.1, 0.15, 0.08, 0.2, 0.1)
    }
    print_comparison_table(results, 'Baseline')

def test_train_and_measure_retrocompatibility():
    # Deve ser retrocompatível (idêntico ao teste anterior sem oráculo)
    res1 = train_and_measure(
        model_class=PredictiveCodingNetwork,
        model_kwargs={'layer_sizes': [7, 8, 2]},
        epochs=1,
        seed=42,
        task_gate_fn=None
    )
    res2 = train_and_measure(
        model_class=PredictiveCodingNetwork,
        model_kwargs={'layer_sizes': [7, 8, 2]},
        epochs=1,
        seed=42
    )
    assert res1 == res2
