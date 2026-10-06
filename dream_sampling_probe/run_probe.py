import numpy as np
from integration_v4.run_integration_experiment import (
    run_experiment,
    sample_dream_cell_action,
    cell_to_obs,
    get_region_transitions,
    calc_mse,
    rejection_sample_reset
)
from dual_weight.dual_weight_pcn import DualWeightPCN
from active_inference.agent import ActiveInferenceAgent
from active_inference.learning_progress import LearningProgressTracker
from active_inference.environment import GridWorld

def run_experiment_accumulated(gamma_importance=0.9, beta_slow=0.00005, beta_delta=0.1, threshold=0.015, num_eps_per_region=225):
    np.random.seed(42)
    env = GridWorld(size=5, goal=(0,0))
    transitions_A = get_region_transitions(env, 'A')
    transitions_B = get_region_transitions(env, 'B')

    dual_weight = DualWeightPCN(layer_sizes=[7, 16, 8, 2], beta_slow=beta_slow, beta_delta=beta_delta,
                                inference_lr=0.1, inference_steps=10, weight_lr=0.05, seed=42)

    tracker = LearningProgressTracker(n_cells=25, n_actions=5)
    agent = ActiveInferenceAgent(world_model=dual_weight.fast, goal_obs=np.array([0.0, 0.0]), tracker=tracker)

    was_above_threshold = False
    cooldown_counter = 0
    protecting = False
    importance_total = None  # None até o primeiro changepoint (mesma semântica inicial do original)

    def run_episodes(num_eps, region):
        nonlocal was_above_threshold, cooldown_counter, protecting, importance_total
        for ep in range(num_eps):
            obs = rejection_sample_reset(env, region)
            for step in range(30):
                action = agent.select_action(obs, ep)
                next_obs, done, success = env.step(action)

                a_onehot = np.zeros(5, dtype=np.float32)
                a_onehot[action] = 1.0
                x_input = np.concatenate([obs, a_onehot])

                res = dual_weight.step(x_input, next_obs)

                cell = int(round(obs[0]*4)*5 + round(obs[1]*4))
                agent.tracker.update(cell, action, res['energia_fast_train_step'])

                if cooldown_counter > 0:
                    cooldown_counter -= 1

                smoothed_delta = res['smoothed_delta']
                if smoothed_delta > threshold and not was_above_threshold and cooldown_counter == 0:
                    snapshot_atual = agent.tracker.visit_count.copy()
                    if importance_total is None:
                        importance_total = snapshot_atual.astype(float)
                    else:
                        importance_total = gamma_importance * importance_total + snapshot_atual.astype(float)
                    protecting = True
                    was_above_threshold = True
                    cooldown_counter = 100
                elif smoothed_delta <= threshold:
                    was_above_threshold = False

                if protecting and importance_total is not None:
                    for _ in range(4):
                        cell_dream, action_dream = sample_dream_cell_action(importance_total, np.random)
                        obs_dream = cell_to_obs(cell_dream)
                        ad_onehot = np.zeros(5, dtype=np.float32)
                        ad_onehot[action_dream] = 1.0
                        x0_dream = np.concatenate([obs_dream, ad_onehot])
                        y_dream = dual_weight.slow.predict(x0_dream)
                        dual_weight.fast.train_step(x0_dream, y_dream)

                obs = next_obs
                if success:
                    break

    run_episodes(num_eps_per_region, 'A')
    erro_A_pos_A = calc_mse(dual_weight.fast, transitions_A)

    env.goal = (4,4)
    agent.goal_obs = np.array([1.0, 1.0])
    run_episodes(num_eps_per_region, 'B')
    erro_A_pos_B = calc_mse(dual_weight.fast, transitions_A)
    erro_B_pos_B = calc_mse(dual_weight.fast, transitions_B)

    env.goal = (0,4)
    agent.goal_obs = np.array([0.0, 1.0])
    transitions_C = get_region_transitions(env, 'C')
    run_episodes(num_eps_per_region, 'C')
    erro_A_pos_C = calc_mse(dual_weight.fast, transitions_A)
    erro_B_pos_C = calc_mse(dual_weight.fast, transitions_B)

    return erro_A_pos_A, erro_A_pos_B, erro_B_pos_B, erro_A_pos_C, erro_B_pos_C

def calc_red(base_err, exp_err):
    return (base_err - exp_err) / base_err

if __name__ == '__main__':
    print("Rodando Sem Sono...")
    res_sem = run_experiment(sleep_enabled=False)
    
    print("Rodando Com Sono (politica original, Fase 9)...")
    res_com_orig = run_experiment(sleep_enabled=True)
    
    print("Rodando Com Sono (acumulador novo)...")
    res_com_acc = run_experiment_accumulated(gamma_importance=0.9)

    print("\n--- RESULTADOS ---")
    print("Model            | A pos A  | A pos B  | B pos B  | A pos C  | B pos C  | Red A-pos-C  | Red B-pos-C ")
    print("-" * 105)

    def print_line(name, res, base_res):
        A_A, A_B, B_B, A_C, B_C = res
        if base_res is None:
            red_A = "-"
            red_B = "-"
        else:
            red_A_val = calc_red(base_res[3], A_C) * 100
            red_B_val = calc_red(base_res[4], B_C) * 100
            red_A = f"{red_A_val:>10.2f}%"
            red_B = f"{red_B_val:>10.2f}%"
        print(f"{name:<16} | {A_A:.4f}   | {A_B:.4f}   | {B_B:.4f}   | {A_C:.4f}   | {B_C:.4f}   | {red_A:>12} | {red_B:>11}")

    print_line("Sem Sono", res_sem, None)
    print_line("Com Sono (Orig)", res_com_orig, res_sem)
    print_line("Com Sono (Acc)", res_com_acc, res_sem)
