import argparse
import torch
import mlflow
import numpy as np
import gymnasium as gym

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
    max_episode_steps: int = 800,
) -> Tuple[bool, float]:
    """Train a reinforcement learning agent in a gym environment.

    .. Args::
        - n_epochs (int): Number of epochs to train the agent.
        - max_episode_steps (int): Number of actions available in the environment.
        - device (torch.device): Device used for training.

    .. Return::
        - None
    """
    epochs_ = tqdm(range(epochs), total=epochs, desc='Training Agent')
    moving_score = deque(maxlen=100)

    for epoch in epochs_:
        score = 0
        state, _ = env.reset()
        for _ in range(max_episode_steps):
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

        mlflow.log_metric('metric.total_score', score, step=epoch)
        mlflow.log_metric('metric.average_score', np.mean(moving_score), step=epoch)
        if isinstance(loss, float):
            mlflow.log_metric('loss', loss, step=epoch)

        if average_score >= 200.0:
            print(f'[INFO] Environment solved! Training done in {epoch} epochs.')
            return True, epoch

    print(f'Environment could not be solved within {epochs} epochs.')
    return False, float('inf')


def evaluate_agent(agent, env: gym.Env, epochs: int = 20, max_episode_steps: int = 500) -> np.float64:
    """Evaluate a trained agent on a given environment"""
    epochs_ = tqdm(range(epochs), total=epochs, desc='Evaluation')
    scores = []

    for epoch in epochs_:
        score = 0
        state, _ = env.reset()
        for _ in range(max_episode_steps):
            action = agent(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            state = new_state
            score += reward
            if terminated or truncated:
                break

        scores.append(score)
        epochs_.set_postfix(current_score=f'{score:.2f}')
        mlflow.log_metric('total_score_eval', score, step=epoch)

    mean_score = np.mean(scores)
    mlflow.log_metric('eval_score', mean_score)

    return mean_score


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-e', '--epochs', type=int, default=1, required=False)
    argparser.add_argument('-d', '--dims', type=tuple, default=(128, 128, 64), required=False)
    argparser.add_argument('-D', '--device', type=str)
    argparser.add_argument('-L', '--log', type=bool, default=True)
    argparser.add_argument('--log-system-metrics', action=argparse.BooleanOptionalAction)
    argparser.add_argument('--experiment-name', type=str, default='LunarLander-v3')
    argparser.add_argument('--uri', type=str, default=None)
    args = argparser.parse_args()

    # initialize
    if args.device is not None:
        device = args.device
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env = gym.make(args.experiment_name)
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

        # model params
        model_input, _ = env.reset()
        model_output = np.empty([env.action_space.n], dtype=np.float32)
        mlflow.log_param('net.type', agent.qnet_target.model_type)
        mlflow.log_param('net.param_count', agent.qnet_target.parameter_count)
        mlflow.log_param('net.dims', args.dims)
        mlflow.log_param('net.layers', len(args.dims))
        mlflow.log_param('net.device', device)

        # environment params
        mlflow.set_tag('environment', args.experiment_name)
        mlflow.log_param('epochs', args.epochs)
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


        # plot figure
        df = unpack_metric_histories(client=client, run_id=run.info.run_id, keys=('total_score',))
        figure = plot_loss_curve(df)
        mlflow.log_figure(figure=figure, artifact_file='summary.png')

        # Save the agent and its model params only if it solves the environment and achieves a higher score than the
        # previous agents
        if code:
            signature = infer_signature(model_input.astype(np.float32), model_output)
            m_local = mlflow.pytorch.log_model(
                agent.qnet_local, name='qnet_local', model_type=agent.qnet_local.model_type, signature=signature
            )
            m_target = mlflow.pytorch.log_model(
                agent.qnet_target, name='qnet_target', model_type=agent.qnet_local.model_type, signature=signature
            )
            eval_score = evaluate_agent(agent=agent, env=env)
            runs = mlflow.search_runs(
                run.info.experiment_id, filter_string=f'metrics.eval_score > {eval_score:.2f}',
                order_by=['metrics.eval_score'], search_all_experiments=False,
            )
            if len(runs) == 0:
                result = mlflow.register_model(m_local.model_uri, 'qnet_local')
                result = mlflow.register_model(m_target.model_uri, 'qnet_target')
            else:
                print(f'The following runs already contain better models: {runs}')

        mlflow.end_run()
    env.close()

    exit(code=not code)
