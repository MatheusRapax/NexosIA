from integration_v8.run_integration_experiment import get_region_transitions
from active_inference.environment import GridWorld

def get_standard_transitions(size=5, goal=(0,0)):
    env = GridWorld(size=size, goal=goal)
    transitions_A = get_region_transitions(env, 'A')
    transitions_B = get_region_transitions(env, 'B')
    transitions_C = get_region_transitions(env, 'C')
    return transitions_A, transitions_B, transitions_C
