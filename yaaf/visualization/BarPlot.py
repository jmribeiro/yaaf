import matplotlib.pyplot as plt
import numpy as np

from yaaf.visualization import confidence_interval


def confidence_bar_plot(names, means, N, title, y_label, confidence=0.95):
    confidence_intervals = [confidence_interval(means[i], N[i], confidence) for i in range(len(means))]
    fig, ax = plt.subplots()
    x_pos = np.arange(len(names))
    ax.bar(x_pos, means, yerr=confidence_intervals, align='center', alpha=0.5, color="green", ecolor='black', capsize=10)
    ax.set_ylabel(y_label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names)
    ax.set_title(title)
    ax.yaxis.grid(True)
    plt.tight_layout()

