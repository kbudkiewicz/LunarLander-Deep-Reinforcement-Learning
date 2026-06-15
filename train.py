import argparse
import gym
import torch
import mlflow
import numpy as np

# Standard library
from collections import deque
from typing import Optional, Tuple, Union

# Libraries
# External
from mlflow.tracking import MlflowClient
from tqdm import tqdm

# Internal
from agent import Agent
from nn import FeedForwardNetwork
from plotting import plot_loss_curve, unpack_metric_histories
from utils import get_nvml_info, get_git_info, get_module_info
from mlflow.models.signature import infer_signature


def train(
    agent,
    env: gym.Env,
    epochs: int,
    n_actions: int = 800,
) -> Tuple[bool, float]:
    """Train a reinforcement learning agent in a gym environment.

    .. Args::
        - n_epochs (int): Number of epochs to train the agent.
        - n_actions (int): Number of actions available in the environment.
        - device (torch.device): Device used for training.

    .. Return::
        - None
    """
    mlflow.log_param('environment', 'LunarLander-v2')
    print('Environment initialized.')

    epochs_ = tqdm(range(epochs), total=epochs, desc='Training Agent')
    moving_score = deque(maxlen=100)

    for epoch in epochs_:
        score = 0
        state, _ = env.reset()
        for _ in range(n_actions):
            action = agent(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            loss = agent.memorize(state, action, reward, next_state, terminated)
            state = next_state
            score += reward
            if terminated or truncated:
                break
            agent.update_epsilon(epoch=epoch)

        moving_score.append(score)
        average_score = np.mean(moving_score)
        epochs_.set_postfix(average_score=f'{average_score:.2f}')

        mlflow.log_metric('total_score', score, step=epoch)
        mlflow.log_metric('average_score', np.mean(moving_score), step=epoch)
        if isinstance(loss, float):
            mlflow.log_metric('loss', loss, step=epoch)

        if average_score >= 200.0:
            print(f'[INFO] Environment solved! Training done in {epoch} epochs.')
            env.close()
            return True, epoch

    print(f'Environment could not be solved within {epochs} epochs.')
    env.close()
    return False, float('inf')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-e', '--epochs', type=int, default=1, required=False)
    argparser.add_argument('-d', '--dims', type=tuple, default=(128, 128, 64), required=False)
    argparser.add_argument('-D', '--device', type=str)
    argparser.add_argument('-L', '--log', type=bool, default=True)
    argparser.add_argument('--log-system-metrics', action=argparse.BooleanOptionalAction)
    argparser.add_argument('--experiment-name', type=str, default='LunarLander_v2')
    argparser.add_argument('--uri', type=str, default=None)
    args = argparser.parse_args()

    # initialize
    if args.device is not None:
        device = args.device
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env = gym.make('LunarLander-v2')
    dims = (*env.observation_space.shape, *args.dims, env.action_space.n)
    model = FeedForwardNetwork(*dims, device=device)
    agent = Agent(*dims, device=device)

    if args.log:
        # check server connection
        if isinstance(args.uri, str):
            mlflow.set_tracking_uri(args.uri)
        if mlflow.active_run():
            print("[INFO] Active run detected. Ending run...")
            mlflow.end_run()

        mlflow.set_experiment(args.experiment_name)
        run = mlflow.start_run(log_system_metrics=args.log_system_metrics)
        client = MlflowClient()

        if args.log_system_metrics:
            mlflow.enable_system_metrics_logging()
            mlflow.log_params(get_nvml_info())
        mlflow.log_param('epochs', args.epochs)
        mlflow.log_param('net.type', agent.qnet_target.name)
        mlflow.log_param('net.param_count', agent.qnet_target.parameter_count)
        mlflow.log_param('net.dims', args.dims)
        mlflow.log_param('net.layers', len(args.dims))
        mlflow.log_param('net.device', device)
        mlflow.log_param('agent.type', agent.agent_type)

        mlflow.set_tags(get_module_info())
        mlflow.set_tags(get_git_info())

    try:
        code, epochs = train(agent=agent, env=env, epochs=args.epochs)
    except KeyboardInterrupt:
        mlflow.log_param('KeyboardInterrupt', True)
        epochs = 0
        code = False

    if args.log:
        mlflow.log_param('training_successful', code)
        mlflow.log_param('total_epochs', epochs)

        # save models and their params only if the environment was solved
        if code:
            mlflow.pytorch.log_model(agent.qnet_local, name=agent.qnet_local.name, model_type='dqn')
            mlflow.pytorch.log_model(agent.qnet_target, name=agent.qnet_target.name, model_type='dqn')

        # plot figure
        df = unpack_metric_histories(client=client, run_id=run.info.run_id, keys=('total_score',))
        figure = plot_loss_curve(df)
        mlflow.log_figure(figure=figure, artifact_file='figures/summary.png')

        mlflow.end_run()

    exit(code=not code)
