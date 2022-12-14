# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (C) Les solutions géostack, Inc - All Rights Reserved
#
# This file is part of geostackai
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
#
# https://github.com/geo-stack/geostackai
# =============================================================================
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from matplotlib.figure import Figure

# ---- Standard imports
import json

# ---- Third party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_loss(filename: str) -> pd.DataFrame:
    """
    Get loss values from a metrics.json file.

    Parameters
    ----------
    filename : str
        A path to a metrics.json file.

    Returns
    -------
    df : pd.DataFrame
        A pandas dataframe containing the loss values.

    """
    with open(filename, 'r') as txtfile:
        lines = txtfile.readlines()
    data = [json.loads(line) for line in lines]
    df = pd.DataFrame(data)
    df = df.set_index('iteration')
    return df


def plot_loss(filename: str, title: str = None,
              rolling_window: int = None) -> Figure:
    """
    Plot total loss vs the number of iterations.

    Parameters
    ----------
    filename : str
        A path to a metrics.json file.
    title: str
        The title of the figure.
    rolling_window: int
        Size of the moving window to apply rolling window calculations on
        the loss data.
    Returns
    -------
    Figure
        A matplotlib figure.
    """
    df = get_loss(filename)

    total_loss = df['total_loss'].dropna()
    val_total_loss = df['val_total_loss'].dropna()
    val_loss_cls = df['val_loss_cls'].dropna()

    if rolling_window is not None:
        total_loss = total_loss.rolling(
            rolling_window, center=True).mean()
        val_total_loss = val_total_loss.rolling(
            rolling_window, center=True).mean()
        val_loss_cls = val_loss_cls.rolling(
            rolling_window, center=True).mean()

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(total_loss, label='total loss')
    ax.plot(val_total_loss, '-.', color='orange', ms=3, label='val total loss')
    ax.plot(val_loss_cls, '-.', color='green', ms=3, label='val loss class')

    ax.set_xlabel('Iterations', labelpad=10)
    ax.set_ylabel('Loss', labelpad=10)
    ax.axis(ymin=0, ymax=np.ceil(np.max(total_loss)))
    ax.axis(ymax=1)

    if title is not None:
        fig.suptitle(title)

    ax.legend()
    ax.grid(color='lightgrey')

    fig.tight_layout()

    fig.data = df

    return fig


if __name__ == "__main__":
    filename = (
        "G:/Shared drives/2_PROJETS/211209_CTSpec_AI_inspection_conduites/"
        "2_TECHNIQUE/6_TRAITEMENT/1_DATA/Training/Models/metrics.json"
    )
    df = get_loss(filename)
    fig = plot_loss(filename)
