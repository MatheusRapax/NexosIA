import sys
import os

from probe_harness.train_and_measure import train_and_measure
from probe_harness.report import print_comparison_table
from pcn_core.model import PredictiveCodingNetwork
from gating_local.gated_pcn import GatedPredictiveCodingNetwork
from gating_local.gating_tracker import GatingLocalTracker
from mechanism_i_probe.fixed_allocation_pcn import FixedAllocationPCN

def run_probe():
    LAYER_SIZES = [7, 64, 32, 2]
    GATE_FRAC = 1.0 / 3.0

    results = {}
    print("Running Baseline...")
    results['Baseline'] = train_and_measure(PredictiveCodingNetwork, model_kwargs={'layer_sizes': LAYER_SIZES})
    
    print("Running Gating-dinamico...")
    results['Gating-dinamico'] = train_and_measure(GatedPredictiveCodingNetwork, model_kwargs={'layer_sizes': LAYER_SIZES},
        tracker_class=GatingLocalTracker, tracker_kwargs={'layer_sizes': LAYER_SIZES, 'gate_frac': GATE_FRAC, 'seed': 42})
        
    print("Running Mecanismo I...")
    results['Mecanismo I'] = train_and_measure(FixedAllocationPCN, model_kwargs={'layer_sizes': LAYER_SIZES},
        task_gate_fn=lambda net, task: net.set_task_gate(task))

    print("\n")
    print_comparison_table(results, baseline_name='Baseline')

if __name__ == '__main__':
    run_probe()
