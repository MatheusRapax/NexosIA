import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from run_integration_experiment import run_experiment

res_a_scaled = run_experiment(layer_sizes=[7,64,32,2], sleep_enabled=False, gating_enabled=False)
print(f"res_a_scaled: {res_a_scaled}")
print(f"A_pos_A: {res_a_scaled[0]} <= 0.06996 ? {res_a_scaled[0] <= 0.06996}")
print(f"B_pos_B: {res_a_scaled[2]} <= 0.05928 ? {res_a_scaled[2] <= 0.05928}")
