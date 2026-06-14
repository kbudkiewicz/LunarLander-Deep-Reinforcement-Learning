import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Iterable


def unpack_metric_histories(client: mlflow.MlflowClient, run_id: str, keys: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame({
        key: [i.value for i in client.get_metric_history(run_id, key)] for key in keys
    })


def plot_loss_curve(df: pd.DataFrame, window: int = 100, show: bool = False) -> plt.figure:
    print('[INFO] Plotting loss curve...')
    fig, ax = plt.subplots()


    # calculate the moving mean score and std
    moving_mean_score = df['total_score'].rolling(window=window, min_periods=0).mean()
    moving_std = df['total_score'].rolling(window=window, min_periods=0).std()
    fill_upper = moving_mean_score + moving_std
    fill_bottom = moving_mean_score - moving_std
    ax.plot(moving_mean_score)
    ax.fill_between(df.index, fill_upper, fill_bottom, alpha=0.2, label='_nolegend_')

    ax.set_xlabel('Epochs [#]')
    ax.set_ylabel('Moving Mean Score (window=100)')
    plt.grid()

    if show:
        plt.show()
    else:
        return fig

