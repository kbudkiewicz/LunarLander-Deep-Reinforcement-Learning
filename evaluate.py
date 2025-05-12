import gym
from agent import Agent


def evaluate(*paths,
             visualize: bool = True,
             play_time: int = 1000):
    """
    Run the agent in evaluation mode. No training occurs when this method is called. The agent is allowed to interact
        with the environment for one game only.

    Args:
        paths (Iterable[str, str]): Path to the saved models parameters
        visualize (bool): Whether to visualize the environment while the agent interacts with it
        play_time (int): allowable number of actions per evaluation
    """
    print('Evaluating agent...')
    env = gym.make('LunarLander-v2', render_mode='human' if visualize else None)
    print(f'Environment {env.spec.name} initialized.')
    agent = Agent()
    agent.eps = 0   # prevent the agent from random behaviour
    score = 0

    # WARNING: if network variables are named differently in the dictionary and code then an error occurs
    # -> same variable names so import can work properly
    assert len(paths) == 2, 'Only two paths are supported.'
    agent.load_state_dict(*paths)
    agent.qnet_target.eval()
    agent.qnet_local.eval()

    state, _ = env.reset()
    for _ in range(play_time):
        action = agent(state)
        new_state, reward, terminated, truncated, _ = env.step(action)
        state = new_state
        score += reward
        if terminated or truncated:
            print(f'Final score: {score}')
            break


if __name__ == '__main__':
    evaluate('./model_params/local.pt', './model_params/target.pt')
