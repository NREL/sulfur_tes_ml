import matplotlib.pyplot as plt
import datetime
import os
import numpy as np
from itertools import zip_longest

plt.rcParams["figure.dpi"] = 200
plt.style.use('ggplot')

def plot_results(df, target=None, x="flow-time", scenario_features=["Tw", "Ti"]):
    figures = {}
    for idx, grp in df.groupby(scenario_features):
        if target is None:
            if 'h_hat' in df:
                target = 'h'
            elif 'Tavg_hat' in df:
                target = 'Tavg'
        if target in df:
            ax = grp.plot(x=x, y=target, c='DarkBlue', linewidth=2.5, label="Expected", figsize=(6,4))
        else:
            ax = grp.plot(x=x, y=target+'_hat', c='DarkOrange', linewidth=2.5, label="Predicted", figsize=(6,4))
        if target == 'h':
            title = 'Predicting h for '
            key = 'h_'
            plt.ylabel(r'Heat Transfer Coefficient $(\frac{W}{m^{2}K})$')
            ax.set_xlim(left=1)
            ax.set_ylim(0,100)
        elif target == 'Tavg':
            title = 'Predicting T for '
            key = 'T_'
            plt.ylabel(r'Sulfur Average Temperature $(K)$')
        if target in df:
            plot = grp.plot(x=x, y=target+'_hat', c='DarkOrange', linewidth=2.5, label="Predicted", ax=ax, figsize=(6,4))
        
        for i, sf in enumerate(scenario_features):
            key += f'{scenario_features[i]}_{idx[i]}'
            title += f'{sf} = {idx[i]}'
            if i != len(scenario_features) - 1:
                title += ' '
                key += '_'
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.show()
        fig = ax.get_figure()
        figures[key] = fig
    return figures

def plot_average_error(df, target='Tavg', x="flow-time", x_max=7200, x_stepsize=0.1, scenario_features=["Tw", "Ti"]):
    ax = plt.figure(figsize=(6,4), dpi = 200).add_axes([0,0,1,1])

    count = 0
    n_steps = int(x_max/x_stepsize)
    avg_err = [0] * n_steps
    for idx, grp in df.groupby(scenario_features):
        ax.plot(grp[x], abs(grp[target+'_hat'] - grp[target]), color='black', linewidth=1, alpha=0.5)
        this_err = abs(grp[target+'_hat'] - grp[target]).tolist()
        avg_err = [sum(n) for n in zip_longest(this_err, avg_err, fillvalue=0)] 
        count += 1
        
    avg_err[:] = [n / count for n in avg_err]
    
    avg_err_x = np.arange(0, x_max, x_stepsize)
    ax.plot(avg_err_x, avg_err[:n_steps], color='r', linewidth=5)
    if target == 'h':
        ax.set_xlim(1)
        ax.set_ylim(0,10)
        plt.ylabel(r'Heat Transfer Coefficient $(\frac{W}{m^{2}K})$')
    elif target == 'Tavg':
        ax.set_xlim(left=0)
        plt.ylabel("Temperature (K)")
    else:
        plt.ylabel(target)
    if x=='flow-time':
        plt.title('Average Error vs Time')
        plt.xlabel("Time (s)")
    else:
        plt.title('Average Error')
        plt.xlabel(x)
    
    plt.show()
    
    figure = ax.get_figure()
    
    return figure

def save_figures(figures):
    figures_directory = '../figures/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'
    os.mkdir(figures_directory)
    for key, figure in figures.items():
        figure.savefig(figures_directory + key + '.png', bbox_inches='tight')
    return figures_directory