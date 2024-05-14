import matplotlib.pyplot as plt


def plot_style(figsize=(12, 6), labelsize=20, titlesize=24, ticklabelsize=14, **kwargs):
    basic_style = {
        "figure.figsize": figsize,
        "axes.labelsize": labelsize,
        "axes.titlesize": titlesize,
        "xtick.labelsize": ticklabelsize,
        "ytick.labelsize": ticklabelsize,
        "axes.spines.top": False,
        "axes.spines.bottom": True,
        "axes.spines.right": False,
        "axes.spines.left": False,
    }
    basic_style.update(kwargs)
    return plt.rc_context(rc=basic_style)
