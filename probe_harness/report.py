def calc_red(base_val, exp_val):
    return (base_val - exp_val) / base_val

def print_comparison_table(results: dict, baseline_name: str):
    base = results[baseline_name]
    print(f"{'Model':<20} | {'A pos A':<8} | {'A pos B':<8} | {'B pos B':<8} | {'A pos C':<8} | {'B pos C':<8} | {'Red A-pos-C':<12} | {'Red B-pos-C':<12}")
    print("-" * 100)
    for name, res in results.items():
        A_A, A_B, B_B, A_C, B_C = res
        if name == baseline_name:
            red_a, red_b = "-", "-"
        else:
            red_a = f"{calc_red(base[3], A_C)*100:.2f}%"
            red_b = f"{calc_red(base[4], B_C)*100:.2f}%"
        print(f"{name:<20} | {A_A:.4f}   | {A_B:.4f}   | {B_B:.4f}   | {A_C:.4f}   | {B_C:.4f}   | {red_a:>12} | {red_b:>12}")
