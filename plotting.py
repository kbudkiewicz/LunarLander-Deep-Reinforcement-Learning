import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TODO: create scripts that take data saved in MLFlow, create a plot and a table that can be then saved in there too
def plot_loss_curve(x: dict) -> plt.figure:
    df = pd.DataFrame.from_dict(x)

    fig, ax = plt.subplots()

    return fig
