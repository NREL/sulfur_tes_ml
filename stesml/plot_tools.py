import math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def get_plot_data(y_hat, test_df, test_index, is_recurrent):
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
        test_df["Tc_hat"] = y_hat_plotting
    else:
        test_df["Tc_hat"] = y_hat
    
    return test_df

def plot_test_results(test_df, model_type, target='T'):
    for idx, grp in test_df.groupby(["Tw", "Ti"]):
        if target == 'T':
            ax = grp.plot(x="Time", y='Tc', c='DarkBlue', linewidth=2.5, label="Expected")
        elif target == 'h':
            ax = grp.plot(x="Time", y='h', c='DarkBlue', linewidth=2.5, label="Expected")
        else:
            print('Target must be T or h.\n')
            return None
        plot = grp.plot(x="Time", y='Tc_hat', c='DarkOrange', linewidth=2.5, label="Predicted ({model_type})".format(model_type=model_type), ax=ax)
        plt.title('Tw = {Tw}  Ti = {Ti}'.format(Tw=idx[0], Ti=idx[1]))
        plt.show()

def plot_test_results_sulfur(test_df, model_type, target='T'):
    for idx, grp in test_df.groupby(["Tw", "Ti"]):
        if target == 'T':
            ax = grp.plot(x="flow-time", y='Tavg', c='DarkBlue', linewidth=2.5, label="Expected")
        elif target == 'h':
            ax = grp.plot(x="flow-time", y='h', c='DarkBlue', linewidth=2.5, label="Expected")
            ax.set_xscale('log')
            ax.set_xlim(.1,7200)
            ax.set_yscale('log')
        else:
            print('Target must be T or h.\n')
            return None
        plot = grp.plot(x="flow-time", y='Tc_hat', c='DarkOrange', linewidth=2.5, label="Predicted ({model_type})".format(model_type=model_type), ax=ax)
        plt.title('Tw = {Tw}  Ti = {Ti}'.format(Tw=idx[0], Ti=idx[1]))
        plt.show()
        
def plot_average_error(test_df):
    ax = plt.figure(figsize=(10,5), dpi = 200).add_axes([0,0,1,1])
    
    count = 0
    for idx, grp in test_df.groupby(["Tw", "Ti"]):
        ax.plot(grp['Time'], abs(grp['Tc_hat'] - grp['Tc']), color='black', linewidth=1, alpha=0.5)
        if count == 0:
            avg_err = abs(grp['Tc_hat'] - grp['Tc'])
        else:
            avg_err += abs(grp['Tc_hat'] - grp['Tc'])
        count += 1
        
    avg_err /= count
    
    ax.plot(grp['Time'], avg_err, color='r', linewidth=5)
    #ax.set_ylim(bottom=0,top=20)
    ax.set_xlim(left=0)
    plt.title('Average Error at Time Step t')
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (K)")
    plt.show()

def plot_average_error_sulfur(test_df):
    ax = plt.figure(figsize=(10,5), dpi = 200).add_axes([0,0,1,1])
    
    count = 0
    for idx, grp in test_df.groupby(["Tw", "Ti"]):
        ax.plot(grp['flow-time'], abs(grp['Tc_hat'] - grp['Tavg']), color='black', linewidth=1, alpha=0.5)
        if count == 0:
            avg_err = abs(grp['Tc_hat'] - grp['Tavg'])
        else:
            avg_err += abs(grp['Tc_hat'] - grp['Tavg'])
        count += 1
        
    avg_err /= count
    
    ax.plot(grp['flow-time'], avg_err, color='r', linewidth=5)
    #ax.set_ylim(bottom=0,top=20)
    ax.set_xlim(left=0)
    plt.title('Average Error at Time Step t')
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (K)")
    plt.show()


def plot_error_vs_T(test_df):
    # This function plots the average difference of Tc-Tc_hat
    # according to Tc. To make this meaningful, the x-axis is
    # Tc, but each Tc is rounded down to the nearest integer.
    # If this wasn't done, you would likely only end up with one
    # datapoint per x-value, because Tc is a float and there are
    # very few (if any) repeated Tc values.
    
    avg_temp_diff = {}

    # Start out with a difference of 0 Kelvin
    for i in range(300, 601):
        avg_temp_diff[i] = [0,0]

    # Get the difference, add the difference to the dictionary value
    # with the key being the Temperature (Tc) rounded down to the nearest
    # integer (floor). 
    tc_hat_list = test_df['Tc_hat'].tolist()
    for i, T in enumerate(test_df["Tc"]):
        avg_temp_diff[math.floor(T)][0] += tc_hat_list[i] - T
        avg_temp_diff[math.floor(T)][1] += 1

    # Get the average temperature difference &
    # create lists of x & y data for plotting
    temperature = []
    diff = []
    for T, Tdiff_and_count in avg_temp_diff.items():
        if Tdiff_and_count[1] > 0:
            Tdiff_and_count[0] /= Tdiff_and_count[1]
        temperature.append(T)
        diff.append(Tdiff_and_count[0])

    plt.scatter(temperature, diff)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Average Error (K)")
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