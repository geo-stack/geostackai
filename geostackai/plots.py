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
import numpy as np
import matplotlib.pyplot as plt


def plot_loss(filename: str) -> Figure:
    """
    Plot total loss vs the number of iterations.

    Parameters
    ----------
    filename : str
        A path to a metrics.json file.

    Returns
    -------
    Figure
        A matplotlib figure.

    """
    with open(filename, 'r') as txtfile:
        lines = txtfile.readlines()
    data = [json.loads(line) for line in lines]

    iters = np.array([d['iteration'] for d in data])
    total_loss = np.array([d['total_loss'] for d in data])

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(iters, total_loss)

    ax.set_xlabel('iterations', labelpad=10)
    ax.set_ylabel('Total Loss', labelpad=10)

    fig.tight_layout()

    return fig


if __name__ == "__main__":
    plot_loss("D:/Projets/geostack/nmg_detection_mb/Models/"
              "model_lr_0.0003/metrics.json")
