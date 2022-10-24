import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LOAD_DIR = "./log"
SAVE_DIR = "./plot"
# ---------------- NEEDED MODIFICATION ---------------- #
TITLE      = "Swimmer-v2"
SEEDS      = [3, 5]
EXPR_NAMES = [
    "env_Swimmer-v2-stg_pges-lr_0.1-sig_0.3-pop_20-T_1000-K_100-al_0.5-sub_1-opt_bgd-pol_linear-init_zero-obs_no-fit_div_std",
]
X_AXIS     = "total_steps"
Y_AXIS     = "reward_mean"
LIMIT_STEP = 500000
# ----------------------------------------------------- #
COLORS = [
    'red', 'blue', 'green', 'purple', 'orange',
    'brown', 'cyan', 'black', 'yellow', 'gray',
    'pink',
]


def getData(expr_name):
    x, y = np.arange(0, LIMIT_STEP, 100), []
    for seed in SEEDS:
        file   = os.path.join(LOAD_DIR, expr_name, 'seed_{}'.format(seed), 'record.csv')
        result = pd.read_csv(file) 
        xp     = result[X_AXIS]
        yp     = result[Y_AXIS]
        y.append(np.interp(x, xp, yp))
    mean = np.mean(np.array(y), axis=0)
    std  = np.std(np.array(y), axis=0)
    return x, {'mean': mean, 'std': std}


def plotData(xs, ys):
    plt.figure(figsize=(10, 8))
    for i in range(len(EXPR_NAMES)):
        plt.plot(xs[i], ys[i]['mean'], label=EXPR_NAMES[i], color=COLORS[i])
        plt.fill_between(xs[i], ys[i]['mean']-ys[i]['std'], ys[i]['mean']+ys[i]['std'], color=COLORS[i], alpha=0.1)
    plt.grid()
    plt.legend(fontsize=6, fancybox=True, framealpha=0.3)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlabel(X_AXIS, fontsize=6)
    plt.ylabel(Y_AXIS, fontsize=6)

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    plt.savefig(SAVE_DIR + "/{}_{}.png".format(TITLE, Y_AXIS))
    plt.show()


def main():
    # Constraints
    assert len(EXPR_NAMES) < len(COLORS), "Don't have enough colors to plot !"

    # --------------- GET DATA --------------- #
    xs, ys = [], []
    for name in EXPR_NAMES:
        x, y = getData(name)
        xs.append(x)
        ys.append(y)

    # -------------- PLOT DATA --------------- #
    plotData(xs, ys)

if __name__ == '__main__':
    main()
