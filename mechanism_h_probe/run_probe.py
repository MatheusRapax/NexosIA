import sys
import os

from probe_harness.train_and_measure import train_and_measure
from probe_harness.report import print_comparison_table
from pcn_core.model import PredictiveCodingNetwork
from synaptic_pruning_probe.pruning_pcn import LeakyRecyclingPCN
from mechanism_h_probe.confidence_calibrated_pcn import ConfidenceCalibratedPCN

def run_probe():
    LAYER_SIZES = [7, 64, 32, 2]

    results = {}
    print("Running Baseline...")
    results['Baseline'] = train_and_measure(PredictiveCodingNetwork, model_kwargs={'layer_sizes': LAYER_SIZES})
    
    print("Running Poda (Rodada 14)...")
    results['Poda (Rodada 14)'] = train_and_measure(LeakyRecyclingPCN, model_kwargs={
        'layer_sizes': LAYER_SIZES, 'decay_lambda': 0.001, 'recycle_every': 200, 'recycle_frac': 0.05, 'trace_beta': 0.99})
        
    print("Running Mecanismo H...")
    results['Mecanismo H'] = train_and_measure(ConfidenceCalibratedPCN, model_kwargs={
        'layer_sizes': LAYER_SIZES, 'decay_lambda': 0.01, 'beta': 0.99})

    print("\n")
    print_comparison_table(results, baseline_name='Baseline')

if __name__ == '__main__':
    run_probe()
