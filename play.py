import argparse
import torch
import mlflow
import gymnasium as gym

from agent import Agent
from utils import get_agent_class


if __name__ == '__main__':
    argsparer = argparse.ArgumentParser(
        description='Visualize the best agent for a given environment saved in the model registry.'
    )
    argsparer.add_argument('-E', '--environment-id', type=str, required=False, default='LunarLander-v3',
        help='Identifier of the environment'
    )
    argsparer.add_argument('-e', '--epochs', type=int, default=10, help='Number of epoch to play')
    argsparer.add_argument('-s', '--max-episode-steps', type=int, default=500,
        help='Maximal number of allowed steps in the environment per epoch'
    )
    argsparer.add_argument('-D', '--device', type=str)
    argsparer.add_argument('-m', '--model-name', type=str, default='qnet_local')
    argsparer.add_argument('-a', '--agent', type=str, required=True)
    args = argsparer.parse_args()

    if args.device is not None:
        device = args.device
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    try:
        env = gym.make(id=args.environment_id, max_episode_steps=args.max_episode_steps, render_mode='human')
    except ValueError as e:
        raise e

    # get latest model version
    client = mlflow.tracking.MlflowClient()
    latest_version = client.get_latest_versions(name=args.model_name, stages=['None', 'Staging', 'Production'])
    if len(latest_version) == 0:
        raise mlflow.exceptions.MlflowException('No model was found for the specified environment')

    # import the latest model
    latest = max(latest_version, key=lambda x: int(x.version))
    model_uri = f"models:/qnet_local/{latest.version}"
    model = mlflow.pytorch.load_model(model_uri=model_uri, device=device)

    if not isinstance(args.agent, str):
        raise ValueError("Agent must be a string.")
    else:
        AgentClass = get_agent_class(args.agent)
        # TODO: check if the agent fits the chosen environment (continuous or discrete)
        agent = AgentClass(
            model=model, device=device, action_space=env.action_space.n, criterion=None
        )
        agent.zero_epsilon()

    try:
        for _ in range(args.epochs):
            terminated = truncated = False
            state, _ = env.reset()
            while not (truncated or terminated):
                action = agent(state)
                new_state, _, terminated, truncated, _ = env.step(action)
                state = new_state
    except KeyboardInterrupt:
        print("[INFO]: Received KeyboardInterrupt. Game stopped.")

    env.close()
    exit(0)
