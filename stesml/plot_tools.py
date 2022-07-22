import matplotlib.pyplot as plt
import datetime
import os
import numpy as np
from itertools import zip_longest

plt.rcParams["figure.dpi"] = 200
plt.style.use('ggplot')

def plot_results(df, model_type, target='Tavg', x="flow-time", scenario_features=["Tw", "Ti"]):
    figures = {}
    for idx, grp in df.groupby(scenario_features):
        ax = grp.plot(x=x, y=target, c='DarkBlue', linewidth=2.5, label="Expected", figsize=(6,4))
        if target == 'h':
            ax.set_xscale('log')
            ax.set_xlim(0.001,7200)
            ax.set_yscale('log')
        plot = grp.plot(x=x, y=target+'_hat', c='DarkOrange', linewidth=2.5, label="Predicted ({model_type})".format(model_type=model_type), ax=ax, figsize=(6,4))
        title = ''
        key = ''
        for i, sf in enumerate(scenario_features):
            key += f'{idx[i]}'
            title += f'{sf} = {idx[i]}'
            if i != len(scenario_features) - 1:
                title += ' '
                key += '_'
        plt.title(title)
        plt.show()
        fig = ax.get_figure()
        figures[key] = fig
    return figures

def save_figures(figures):
    figures_directory = '../figures/' + datetime.datetime.now().strftime("%Y%m%d-%H%M") + '/'
    os.mkdir(figures_directory)
    for key, figure in figures.items():
        figure.savefig(figures_directory + key + '.png', bbox_inches='tight')
    return figures_directory

def plot_average_error(df, target='Tavg', t_min=-1, t_max=-1, x="flow-time", scenario_features=["Tw", "Ti"]):
    if t_min > 0:
        df = df[df[x] >= t_min]
    if t_max > 0:
        df = df[df[x] <= t_max]
    ax = plt.figure(figsize=(6,4), dpi = 200).add_axes([0,0,1,1])
    
    count = 0
    avg_err = [0] * 72000
    for idx, grp in df.groupby(scenario_features):
        ax.plot(grp[x], abs(grp[target+'_hat'] - grp[target]), color='black', linewidth=1, alpha=0.5)
        this_err = abs(grp[target+'_hat'] - grp[target]).tolist()
        avg_err = [sum(n) for n in zip_longest(this_err, avg_err, fillvalue=0)] 
        count += 1
        
    avg_err[:] = [n / count for n in avg_err]
    
    avg_err_time = np.arange(0, 7200, .1)
    ax.plot(avg_err_time, avg_err[:72000], color='r', linewidth=5)
    if target == 'h':
        ax.set_xlim(1)
        ax.set_ylim(0,10)
        plt.ylabel("Heat Transfer Coefficient (W/((m^2)K)")
    elif target == 'Tavg':
        ax.set_xlim(left=0)
        plt.ylabel("Temperature (K)")
    plt.title('Average Error at Time Step t')
    plt.xlabel("Time (s)")
    
    plt.show()
    
    figure = ax.get_figure()
    
    return figure