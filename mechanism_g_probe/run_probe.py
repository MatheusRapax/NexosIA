import sys
import os

from probe_harness.train_and_measure import train_and_measure
from probe_harness.report import print_comparison_table
from pcn_core.model import PredictiveCodingNetwork
from gating_local.gated_pcn import GatedPredictiveCodingNetwork
from gating_local.gating_tracker import GatingLocalTracker
from mechanism_g_probe.lateral_inhibition_windowed import LateralInhibitionWindowedPCN
from mechanism_g_probe.context_gated_lateral_inhibition import ContextGatedLateralInhibitionPCN

def run_probe():
    layer_sizes = [7, 64, 32, 2]
    seed = 42

    print("Running Baseline...")
    res_baseline = train_and_measure(
        model_class=PredictiveCodingNetwork,
        model_kwargs={'layer_sizes': layer_sizes},
        epochs=50,
        seed=seed
    )

    print("Running Gating-only...")
    res_gating = train_and_measure(
        model_class=GatedPredictiveCodingNetwork,
        model_kwargs={'layer_sizes': layer_sizes},
        tracker_class=GatingLocalTracker,
        tracker_kwargs={'layer_sizes': layer_sizes, 'gate_frac': 0.5, 'seed': seed},
        epochs=50,
        seed=seed
    )

    print("Running G-lite...")
    res_glite = train_and_measure(
        model_class=LateralInhibitionWindowedPCN,
        model_kwargs={'layer_sizes': layer_sizes, 'gamma': 0.01, 'window_steps': 200},
        epochs=50,
        seed=seed
    )

    print("Running G-v2...")
    res_gv2 = train_and_measure(
        model_class=ContextGatedLateralInhibitionPCN,
        model_kwargs={'layer_sizes': layer_sizes, 'gamma': 0.01},
        tracker_class=GatingLocalTracker,
        tracker_kwargs={'layer_sizes': layer_sizes, 'gate_frac': 0.5, 'seed': seed},
        epochs=50,
        seed=seed
    )

    results = {
        'Baseline': res_baseline,
        'Gating-only': res_gating,
        'G-lite': res_glite,
        'G-v2': res_gv2
    }

    print("\n")
    print_comparison_table(results, 'Baseline')

if __name__ == '__main__':
    run_probe()
