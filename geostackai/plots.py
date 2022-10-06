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

    ax.set_xlabel('Iterations', labelpad=10)
    ax.set_ylabel('Total Loss', labelpad=10)
    ax.axis(ymin=0, ymax=np.ceil(np.max(total_loss)))

    fig.tight_layout()

    return fig


if __name__ == "__main__":
    plot_loss("D:/Projets/geostack/ctspec_ai/CTSpecAiDataSet_20220706/"
              "Models/categories_v1/metrics.json")
    plot_loss("D:/Projets/geostack/ctspec_ai/CTSpecAiDataSet_20220706/"
              "Models/supercategories_v1/metrics.json")
