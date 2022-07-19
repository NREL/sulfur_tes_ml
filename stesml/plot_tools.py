import math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_results(df, model_type, target='Tavg'):
    for idx, grp in df.groupby(["Tw", "Ti"]):
        ax = grp.plot(x="flow-time", y=target, c='DarkBlue', linewidth=2.5, label="Expected")
        if target == 'h':
            ax.set_xscale('log')
            ax.set_xlim(0.001,7200)
            ax.set_yscale('log')
        plot = grp.plot(x="flow-time", y=target+'_hat', c='DarkOrange', linewidth=2.5, label="Predicted ({model_type})".format(model_type=model_type), ax=ax)
        plt.title('Tw = {Tw}  Ti = {Ti}'.format(Tw=idx[0], Ti=idx[1]))
        plt.show()

def plot_average_error(df, target='Tavg', t_min=-1, t_max=-1):
    if t_min > 0:
        df = df[df['flow-time'] >= t_min]
    if t_max > 0:
        df = df[df['flow-time'] <= t_max]
    ax = plt.figure(figsize=(10,5), dpi = 200).add_axes([0,0,1,1])
    
    count = 0
    for idx, grp in df.groupby(["Tw", "Ti"]):
        grp = grp.head(72000) # Trim dataset so all groups have same length
        ax.plot(grp['flow-time'], abs(grp[target+'_hat'] - grp[target]), color='black', linewidth=1, alpha=0.5)
        if count == 0:
            avg_err = abs(grp[target+'_hat'] - grp[target])
        else:
            avg_err += abs(grp[target+'_hat'] - grp[target])
        count += 1
        
    avg_err /= count
    
    fts = grp['flow-time'].shape[0]
    aes = avg_err.shape[0]
    
    if fts > aes:
        plot_length = aes
    else:
        plot_length = fts
    
    ax.plot(grp['flow-time'].head(plot_length), avg_err.head(plot_length), color='r', linewidth=5)
    if target == 'h':
        ax.set_xlim(1)
        ax.set_ylim(0,10)
        plt.ylabel("Heat Transfer Coefficient (W/((m^2)K)")
    else:
        ax.set_xlim(left=0)
        plt.ylabel("Temperature (K)")
    plt.title('Average Error at Time Step t')
    plt.xlabel("Time (s)")
    
    plt.show()
    
    return avg_err