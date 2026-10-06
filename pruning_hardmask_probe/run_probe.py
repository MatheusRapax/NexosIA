import numpy as np
from pruning_hardmask_probe.combo_runner import run_scenario

def calc_red(base_err, exp_err):
    return (base_err - exp_err) / base_err

def print_results(results):
    print("Model            | A pos A  | A pos B  | B pos B  | A pos C  | B pos C  | Red A-pos-C  | Red B-pos-C ")
    print("-" * 105)
    
    base_A_pos_C = results["Baseline"][3]
    base_B_pos_C = results["Baseline"][4]

    for name, res in results.items():
        A_pos_A, A_pos_B, B_pos_B, A_pos_C, B_pos_C = res
        if name == "Baseline":
            red_A = "-"
            red_B = "-"
        else:
            red_A_val = calc_red(base_A_pos_C, A_pos_C) * 100
            red_B_val = calc_red(base_B_pos_C, B_pos_C) * 100
            red_A = f"{red_A_val:>10.2f}%"
            red_B = f"{red_B_val:>10.2f}%"
        
        print(f"{name:<16} | {A_pos_A:.4f}   | {A_pos_B:.4f}   | {B_pos_B:.4f}   | {A_pos_C:.4f}   | {B_pos_C:.4f}   | {red_A:>12} | {red_B:>11}")


if __name__ == "__main__":
    epochs = 200
    
    results = {}
    print("Rodando Baseline...")
    _, _, results["Baseline"] = run_scenario(use_poda=False, use_mask=False, epochs=epochs)
    
    print("Rodando Poda Isolada...")
    _, _, results["Poda Isolada"] = run_scenario(use_poda=True, use_mask=False, epochs=epochs)

    print("Rodando Mascara Isolada...")
    _, _, results["Mascara Isolada"] = run_scenario(use_poda=False, use_mask=True, epochs=epochs)

    print("Rodando Combinado (Poda + Mascara)...")
    _, _, results["Combinado"] = run_scenario(use_poda=True, use_mask=True, epochs=epochs)

    print("\n--- RESULTADOS ---")
    print_results(results)
