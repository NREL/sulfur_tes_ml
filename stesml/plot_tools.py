import math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def get_plot_data(y_hat, test_df, test_index, is_recurrent, target='Tavg'):
    # Recurrent data loses one datapoint because of
    # shifting from series to supervised. For plotting purposes,
    # append the last datapoint onto the end of each series. 
    # This data should only be used for plotting purposes,
    # as the final datapoint is not an actual prediction from
    # the model.
    if is_recurrent:
        y_hat_plotting = y_hat.copy()
        points_per_sample = int(len(y_hat)/len(test_index))
        for i in range(len(test_index)):
            insert_index = (i + 1)*(points_per_sample + 1)
            if i == len(test_index) - 1:
                y_hat_plotting = np.append(y_hat_plotting, y_hat_plotting[insert_index - 2])
            else:
                y_hat_plotting = np.insert(y_hat_plotting, insert_index - 1, y_hat_plotting[insert_index - 2])
        test_df[target+'_hat'] = y_hat_plotting
    else:
        test_df[target+'_hat'] = y_hat
    
    return test_df

def plot_test_results(test_df, model_type, target='Tavg'):
    for idx, grp in test_df.groupby(["Tw", "Ti"]):
        ax = grp.plot(x="flow-time", y=target, c='DarkBlue', linewidth=2.5, label="Expected")
        if target == 'h':
            ax.set_xscale('log')
            ax.set_xlim(.1,7200)
            ax.set_yscale('log')
        plot = grp.plot(x="flow-time", y=target+'_hat', c='DarkOrange', linewidth=2.5, label="Predicted ({model_type})".format(model_type=model_type), ax=ax)
        plt.title('Tw = {Tw}  Ti = {Ti}'.format(Tw=idx[0], Ti=idx[1]))
        plt.show()

def plot_average_error(test_df, target='Tavg'):
    ax = plt.figure(figsize=(10,5), dpi = 200).add_axes([0,0,1,1])
    
    count = 0
    for idx, grp in test_df.groupby(["Tw", "Ti"]):
        grp = grp.head(72000) # Trim dataset so all groups have same length
        ax.plot(grp['flow-time'], grp[target+'_hat'] - grp[target], color='black', linewidth=1, alpha=0.5)
        if count == 0:
            avg_err = abs(grp[target+'_hat'] - grp[target])
        else:
            avg_err += abs(grp[target+'_hat'] - grp[target])
        count += 1
        
    avg_err /= count
    
    ax.plot(grp['flow-time'], avg_err, color='r', linewidth=5)
    if target == 'h':
        ax.set_xlim(1)
        ax.set_ylim(-10,10)
        plt.ylabel("Heat Transfer Coefficient (W/((m^2)K)")
    else:
        ax.set_xlim(left=0)
        plt.ylabel("Temperature (K)")
    plt.title('Average Error at Time Step t')
    plt.xlabel("Time (s)")
    
    plt.show()

def plot_progress_results(history, model_type, is_recurrent, metric):
    fig, ax = plt.subplots(figsize=(12,8), dpi= 200, facecolor='w', edgecolor='k')
    
    if is_recurrent:
        model_type = "Recurrent " + model_type
    else:
        model_type = "Non-recurrent " + model_type

    ax.plot(*zip(*history))
    plt.xlabel("n_estimators")
    plt.ylabel(metric)
    plt.title("{model_type} Model: {metric} for Test Data".format(model_type=model_type, metric=metric))
    plt.show()