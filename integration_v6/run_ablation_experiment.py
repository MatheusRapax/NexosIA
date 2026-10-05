import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "active_inference"))

import integration_v6.run_integration_experiment as run_exp
from hard_mask_local.hard_mask_tracker import HardMaskLocalTracker
from hard_mask_local.ablation_zero_tracker import HardMaskZeroTracker

def run_with_tracker(tracker_class, k):
    old_tracker = run_exp.HardMaskLocalTracker
    run_exp.HardMaskLocalTracker = tracker_class
    try:
        res = run_exp.run_experiment(sleep_enabled=True, hm_enabled=True, hm_k_frac=k, hm_gamma=0.9)
        return res
    finally:
        run_exp.HardMaskLocalTracker = old_tracker

def main():
    k_values = [0.1, 0.25, 0.5, 0.75]
    print(f"{'K':<5} | {'B-pos-B (ancorado)':<18} | {'B-pos-B (zerado)':<18} | {'Delta (zerado - ancorado)':<25}")
    for k in k_values:
        res_anchored = run_with_tracker(HardMaskLocalTracker, k)
        res_zeroed = run_with_tracker(HardMaskZeroTracker, k)
        
        b_pos_b_anchored = res_anchored[2]
        b_pos_b_zeroed = res_zeroed[2]
        delta = b_pos_b_zeroed - b_pos_b_anchored
        
        print(f"{k:<5} | {b_pos_b_anchored:<18.4f} | {b_pos_b_zeroed:<18.4f} | {delta:<25.4f}")

if __name__ == '__main__':
    main()
