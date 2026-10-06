import numpy as np
from integration_v8.run_integration_experiment import get_region_transitions, calc_mse
from active_inference.environment import GridWorld
from pcn_core.model import PredictiveCodingNetwork
from synaptic_pruning_probe.pruning_pcn import LeakyRecyclingPCN
from hard_mask_local.hard_mask_tracker import HardMaskLocalTracker

def run_scenario(use_poda, use_mask, epochs, k_frac=0.5, decay_lambda=0.001, recycle_every=200, recycle_frac=0.05, trace_beta=0.99):
    np.random.seed(42)
    env = GridWorld(size=5, goal=(0,0))
    transitions_A = get_region_transitions(env, 'A')
    transitions_B = get_region_transitions(env, 'B')
    transitions_C = get_region_transitions(env, 'C')

    layer_sizes = [7, 64, 32, 2]
    if use_poda:
        net = LeakyRecyclingPCN(layer_sizes=layer_sizes, seed=42, decay_lambda=decay_lambda,
                                 recycle_every=recycle_every, recycle_frac=recycle_frac, trace_beta=trace_beta)
    else:
        net = PredictiveCodingNetwork(layer_sizes=layer_sizes, seed=42)

    mask_tracker = HardMaskLocalTracker(layer_sizes=layer_sizes, k_frac=k_frac, seed=42) if use_mask else None

    def train_on(transitions):
        for _ in range(epochs):
            np.random.shuffle(transitions)
            for obs, a, next_obs in transitions:
                a_onehot = np.zeros(5, dtype=np.float32)
                a_onehot[a] = 1.0
                x_input = np.concatenate([obs, a_onehot])
                net.train_step(x_input, next_obs)
                if mask_tracker is not None:
                    mask_tracker.update_ema(net.last_local_grad)
                    mask_tracker.apply_penalty(net.W)

    train_on(transitions_A)
    if mask_tracker is not None:
        mask_tracker.on_changepoint(net.W)  # fronteira real A->B
    erro_A_pos_A = calc_mse(net, transitions_A)

    train_on(transitions_B)
    erro_A_pos_B = calc_mse(net, transitions_A)
    erro_B_pos_B = calc_mse(net, transitions_B)
    if mask_tracker is not None:
        mask_tracker.on_changepoint(net.W)  # fronteira real B->C

    train_on(transitions_C)
    erro_A_pos_C = calc_mse(net, transitions_A)
    erro_B_pos_C = calc_mse(net, transitions_B)

    return net, mask_tracker, (erro_A_pos_A, erro_A_pos_B, erro_B_pos_B, erro_A_pos_C, erro_B_pos_C)
