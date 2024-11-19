import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc
from ElectrodeDataset import ElectrodeDataset

def visualize_channel_weights(svm_model):
    # load the CSV file containing the x and y coordinates of the electrodes..
    electrode_positions = pd.read_csv('data/BCIsensor_xy.csv', header=None, names=["x", "y"])

    # Since we have 102 electrode locations and 204 channels, each electrode will have two channels.
    svm_weights = svm_model.weights()

    # reshape the weights to match the electrodes - averaging the weights of the two channels for each electrode
    weights_magnitude = np.sqrt(np.sum(svm_weights.reshape(-1, 2)**2, axis=1))

    # create the grid to interpolate
    # The grid will be a bit larger than the electrode positions to ensure all points are within the interpolation area
    grid_x, grid_y = np.mgrid[min(electrode_positions['x']):max(electrode_positions['x']):100j,
                            min(electrode_positions['y']):max(electrode_positions['y']):100j]

    # interpolate the weights onto this grid
    grid_z = griddata(electrode_positions[['x', 'y']].values, weights_magnitude, (grid_x, grid_y), method='cubic')

    # plotting
    plt.figure(figsize=(10, 8))
    plt.imshow(grid_z.T, extent=(min(electrode_positions['x']), max(electrode_positions['x']),
                                min(electrode_positions['y']), max(electrode_positions['y'])),
            origin='lower', cmap='jet')
    plt.colorbar(label='SVM Weight')
    plt.axis('equal')
    plt.axis('off')
    plt.title('Visualization of SVM Weights on the Brain Surface')
    plt.show()

def visualize_channel_stem_plot(svm_model):
    # get the signed weights from the trained SVM model
    svm_weights = svm_model.weights()

    print(svm_model.model.support_vectors_)

    # generate a sequence of channel indices
    channel_indices = np.arange(len(svm_weights))

    # find the indices of the six largest magnitude weights
    dominant_indices = np.argsort(np.abs(svm_weights))[-6:]

    # create the stem plot for all weights
    plt.figure(figsize=(14, 6))
    plt.stem(channel_indices, svm_weights, 'grey', markerfmt=' ', basefmt=" ")

    # highlight the six dominant channels
    plt.stem(channel_indices[dominant_indices], svm_weights[dominant_indices], 'C3', markerfmt='o', basefmt=" ")

    # annotate the six dominant channels
    for i in dominant_indices:
        plt.annotate(f'{i}: {svm_weights[i]:.7f}',
                     (i, svm_weights[i]),
                     textcoords="offset points",
                     xytext=(0,10),
                     ha='center',
                     color='darkred')

    # set the labels and title
    plt.xlabel('Channel Index')
    plt.ylabel('Weight')
    plt.title('Stem Plot of Signed Weights for Each Channel with Dominant Channels Highlighted')

    # show the plot
    plt.show()

    # print the dominant channels and their weights
    print("Dominant Channels and their Weights:")
    for i in dominant_indices:
        print(f'Channel {i}: Weight {svm_weights[i]}')

def plot_electrode_positions():
    # load the CSV file containing the x and y coordinates of the electrodes..
    electrode_positions = pd.read_csv('data/BCIsensor_xy.csv', header=None, names=["x", "y"])

    # scatter plot of the electrode positions
    plt.scatter(electrode_positions['x'], electrode_positions['y'], c='blue', label='Electrodes')

    # connect each sequential electrode with a line
    for i in range(len(electrode_positions) - 1):
        plt.plot(electrode_positions['x'][i:i+2], electrode_positions['y'][i:i+2], c='grey')

    # label each point with the electrode index
    for idx, row in electrode_positions.iterrows():
        plt.text(row['x'], row['y'], str(idx), fontsize=9, ha='right', va='bottom')

    # add labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Electrode Positions with Labels')
    plt.legend()
    plt.show()

def visualize_data_observations(filepath):
    # load the CSV file containing the x and y coordinates of the electrodes..
    electrode_positions = pd.read_csv('data/BCIsensor_xy.csv', header=None, names=["x", "y"])

    # load observations from EEG dataset
    dataset = ElectrodeDataset(filepath, 1).get_data()

    # create a grid for interpolation
    grid_x, grid_y = np.mgrid[min(electrode_positions['x']):max(electrode_positions['x']):100j,
                              min(electrode_positions['y']):max(electrode_positions['y']):100j]

    # plotting
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))  # 2 rows, 5 columns

    for i in range(5):
        # get the ith observation
        observation = dataset[i].values
        
        # calculate the magnitude of the observation for each electrode
        observation_magnitude = np.sqrt(np.sum(observation.reshape(-1, 2)**2, axis=1))

        # interpolate the magnitudes onto this grid
        grid_z = griddata(electrode_positions[['x', 'y']].values, observation_magnitude, (grid_x, grid_y), method='cubic')
        
        # determine subplot indices (row, col)
        row = i // 5
        col = i % 5

        # plot on the corresponding subplot
        im = axs[row, col].imshow(grid_z.T, extent=(min(electrode_positions['x']), max(electrode_positions['x']),
                                                     min(electrode_positions['y']), max(electrode_positions['y'])),
                                  origin='lower', cmap='jet')
        axs[row, col].axis('equal')
        axs[row, col].axis('off')
        axs[row, col].set_title(f'Obs {i+1}')

    # add a colorbar
    fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal', fraction=0.1, pad=0.1, aspect=40, label='Observation Magnitude')
    
    # set a main title for the figure
    plt.suptitle('Visualization of the First 10 Observations on the Brain Surface', fontsize=16)
    plt.show()

def visualize_data_average(filepath):
    # load the CSV file containing the x and y coordinates of the electrodes..
    electrode_positions = pd.read_csv('data/BCIsensor_xy.csv', header=None, names=["x", "y"])

    # load observations from EEG dataset
    dataset = ElectrodeDataset(filepath, 1).get_data()
    
    # Calculate the average across all observations (columns)
    average_vector = dataset.mean(axis=1)
    
    # Calculate the magnitude of the average for each electrode
    # The average_vector is reshaped from 204x1 to 102x2, then magnitude is calculated
    average_magnitude = np.sqrt(np.sum(average_vector.values.reshape(-1, 2)**2, axis=1))
    
    # Create the grid to interpolate
    grid_x, grid_y = np.mgrid[min(electrode_positions['x']):max(electrode_positions['x']):100j,
                              min(electrode_positions['y']):max(electrode_positions['y']):100j]
    
    # Interpolate the magnitudes onto this grid
    grid_z = griddata(electrode_positions[['x', 'y']].values, average_magnitude, (grid_x, grid_y), method='cubic')
    
    # Plotting
    plt.figure(figsize=(10, 8))
    plt.imshow(grid_z.T, extent=(min(electrode_positions['x']), max(electrode_positions['x']),
                                 min(electrode_positions['y']), max(electrode_positions['y'])),
               origin='lower', cmap='jet')
    plt.colorbar(label='Average Observation Magnitude')
    plt.axis('equal')
    plt.axis('off')
    plt.title('Visualization of Average Observation Magnitude on the Brain Surface')
    plt.show()

def visualize_ROC_plots(fpr_list, tpr_list, roc_auc_list):
        plt.figure()
        for i, (fpr, tpr, roc_auc) in enumerate(zip(fpr_list, tpr_list, roc_auc_list)):
            plt.plot(fpr, tpr, lw=2, alpha=0.3,
                     label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
        
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fpr_list, tpr_list)], axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f)' % (mean_auc), lw=2, alpha=0.8)
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.show()
    
if __name__ == '__main__':
    plot_electrode_positions()
    visualize_data_observations("data/feaSubEOvert_1.csv")
    visualize_data_observations("data/feaSubEOvert_2.csv")
    visualize_data_average("data/feaSubEOvert_1.csv")
    visualize_data_average("data/feaSubEOvert_2.csv")