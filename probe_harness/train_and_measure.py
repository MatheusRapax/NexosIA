import numpy as np
from integration_v8.run_integration_experiment import calc_mse
from probe_harness.transitions import get_standard_transitions

def train_and_measure(model_class, model_kwargs=None, tracker_class=None, tracker_kwargs=None, epochs=50, seed=42, task_gate_fn=None):
    np.random.seed(seed)
    transitions_A, transitions_B, transitions_C = get_standard_transitions()

    net = model_class(seed=seed, **(model_kwargs or {}))
    tracker = tracker_class(**(tracker_kwargs or {})) if tracker_class is not None else None
    
    if tracker is not None:
        tracker.apply_gate(net)
    elif task_gate_fn is not None:
        task_gate_fn(net, 'A')

    def signal_boundary(next_task):
        if tracker is not None:
            tracker.on_changepoint()
            tracker.apply_gate(net)
        elif task_gate_fn is not None:
            task_gate_fn(net, next_task)
        elif hasattr(net, 'on_changepoint'):
            net.on_changepoint()

    def train_on(transitions):
        for _ in range(epochs):
            np.random.shuffle(transitions)
            for obs, a, next_obs in transitions:
                a_onehot = np.zeros(5, dtype=np.float32)
                a_onehot[a] = 1.0
                x_input = np.concatenate([obs, a_onehot])
                net.train_step(x_input, next_obs)

    def evaluate(task_name, transitions):
        if task_gate_fn is not None:
            task_gate_fn(net, task_name)
        return calc_mse(net, transitions)

    train_on(transitions_A)
    erro_A_pos_A = evaluate('A', transitions_A)

    signal_boundary('B')
    train_on(transitions_B)
    erro_A_pos_B = evaluate('A', transitions_A)
    erro_B_pos_B = evaluate('B', transitions_B)

    signal_boundary('C')
    train_on(transitions_C)
    erro_A_pos_C = evaluate('A', transitions_A)
    erro_B_pos_C = evaluate('B', transitions_B)

    return erro_A_pos_A, erro_A_pos_B, erro_B_pos_B, erro_A_pos_C, erro_B_pos_C
