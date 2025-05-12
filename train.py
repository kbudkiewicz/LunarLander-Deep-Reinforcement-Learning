import gym
from tqdm import tqdm
from numpy import mean
from collections import deque
from agent import Agent


def train(training_episodes: int = 800, play_time: int = 1000):
    env = gym.make('LunarLander-v2')
    print('Environment initialized.')
    agent = Agent()
    print('Agent initialized.')
    print(f'Current device: {agent.device.upper()}\n')

    metrics = {
        'total_scores': [],
        'total_losses': [],
        'moving_scores': deque(maxlen=100),
        'moving_losses': deque(maxlen=100),
    }

    training_loop = tqdm(range(training_episodes), total=training_episodes, desc='Training Agent')
    for episode in training_loop:
        score = 0
        state, _ = env.reset()
        for _ in range(play_time):
            action = agent(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.memorize(state, action, reward, next_state, terminated)
            state = next_state
            score += reward
            if terminated or truncated:
                break
            agent.update_epsilon(current_episode=episode)

        for k in metrics.keys():
            if 'scores' in k:
                metrics[k].append(score)
            elif 'loss' in k:
                metrics[k].append(agent.loss)
            else:
                raise KeyError(f'Unknown metrics key: {k}')

        avg_score, avg_loss = mean(metrics['moving_scores']), mean(metrics['moving_losses'])
        if episode >= 50 and episode % 50 == 0:
            print(f'\nEpisode #{episode}:\n'
                  f'\tAverage score: {avg_score:.2f}\n'
                  f'\tAverage loss: {avg_loss:.2f}')
            if agent.eps > agent.eps_end:
                print(f'\tEpsilon: {agent.eps:.2f}')

        if avg_score >= 200.0:
            agent.save_state_dict()
            print(f'Environment solved! Training done in {episode} episodes. Average loss: {avg_loss:.2f}')
            env.close()
            break


if __name__ == '__main__':
    train()
