import numpy as np
from rl_mdp.mdp.reward_function import RewardFunction
from rl_mdp.mdp.transition_function import TransitionFunction
from rl_mdp.policy.policy import Policy
from rl_mdp.mdp.mdp import MDP
from rl_mdp.model_free_prediction.monte_carlo_evaluator import MCEvaluator


def main() -> None:
    """
    Starting point of the program, you can instantiate any classes, run methods/functions here as needed.
    """
    states = [0, 1, 2, 3]
    actions = [0, 1]
    rewards = {
        (0,0): 0.0,
        (0,1): 0.0,
        (1,0): 1.0,
        (1,1): -1.0,
        (2,0): 2.0,
        (2,1): -1.0
    }
    reward_function = RewardFunction(rewards)
    transitions = {
        (0,0): np.array([0.0, 0.8, 0.2, 0.0]),
        (0,1): np.array([0.0, 1.0, 0.0, 0.0]),
        (1,0): np.array([0.0, 0.0, 0.5, 0.5]),
        (1,1): np.array([0.0, 1.0, 0.0, 0.0]),
        (2,0): np.array([0.0, 0.0, 0.0, 1.0]),
        (2,1): np.array([0.0, 0.0, 1.0, 0.0]),
    }
    transition_function = TransitionFunction(transitions)
    
    mdp = MDP(states, actions, transition_function, reward_function, discount_factor=0.9)

    num_actions = 2
    policy_mapping = np.array([0, 1])

    policy1 = Policy(policy_mapping, num_actions)
    policy1.set_action_probabilities(0, [0.7, 0.3])
    policy1.set_action_probabilities(1, [0.6, 0.4])
    policy1.set_action_probabilities(2, [0.9, 0.1])

    policy2 = Policy(policy_mapping, num_actions)
    policy2.set_action_probabilities(0, [0.5, 0.5])
    policy2.set_action_probabilities(1, [0.3, 0.7])
    policy2.set_action_probabilities(2, [0.8, 0.2])

    monte_carlo_pc1 = MCEvaluator(mdp)
    monte_carlo_pc2 = MCEvaluator(mdp)

    # Exercise (a)
    for iteration in range(1000):
        state_pc1 = 0
        episode_pc1 = []
        while state_pc1 != 3:
            episode_pc1.append(state_pc1)
            action = policy1.sample_action(state_pc1)
            print(action)
            state_pc1 = mdp.step(policy1.sample_action(state_pc1))
        monte_carlo_pc1.evaluate(episode_pc1)

        state_pc2 = 0
        episode_pc2 = []
        while state_pc2 != 3:
            episode_pc1.append(state_pc2)
            state_pc2 = mdp.step(policy1.sample_action(state_pc2))
        monte_carlo_pc2.evaluate(episode_pc2)

    print(monte_carlo_pc1.value_fun)
    print(monte_carlo_pc2.value_fun)


if __name__ == "__main__":
    main()
