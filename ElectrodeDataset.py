import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class ElectrodeDataset:
    """
    ElectrodeDataset reads EEG data files and provides an interface for the data.
    """

    def __init__(self, filepath, true_class, transform_to_polar=False, normalize=False, use_pca=False):
        """
        ElectrodeDataset constructor.
        
        @param filepath is the path to data.
        @param true_class is the class of the dataset.
        @param transform_to_polar is a flag to use polar coordinates (mag + phase) features.
        """

        self.data = pd.read_csv(filepath, header=None)
        self.true_class = true_class
        self.transform_to_polar = transform_to_polar
        self.normalize = normalize
        self.use_pca = use_pca
        self.n_components = 70

        assert self.data.shape == (204, 120), "Data must be of shape (204, 120)"

        if self.transform_to_polar:
            self.transform_data()
            assert self.data.shape == (204, 120), "Data must be of shape (204, 120)"
        
        if self.use_pca:
            self.data = self.apply_pca()
            assert self.data.shape == (self.n_components, 120), "Data must be of shape (n_components, 120)"

        if self.normalize:
            self.data = self.normalize_data()
            assert self.data.shape == (204, 120), "Data must be of shape (204, 120)"

    def transform_data(self):
        """
        Transforms the data from cartesian coordinates (x, y) to polar coordinates (magnitude, phase).
        The transformation is done in-place with the data structure remaining as 204x120.
        """
        def cartesian_to_polar(x, y):
            """
            Helper function to convert cartesian coordinates to polar coordinates.
            
            :param x: X-coordinate
            :param y: Y-coordinate
            :return: (magnitude, phase)
            """
            magnitude = np.sqrt(x**2 + y**2)
            phase = np.arctan2(y, x)
            return magnitude, phase
        
        # Loop through each column and transform the data
        for col in self.data.columns:
            # Extract the x and y coordinates
            x_coords = self.data[col].iloc[0::2]  # X-coordinates are in the even index positions
            y_coords = self.data[col].iloc[1::2]  # Y-coordinates are in the odd index positions
            
            # Compute magnitude and phase for each pair of coordinates
            mag_phase = [cartesian_to_polar(x, y) for x, y in zip(x_coords, y_coords)]
            mag_phase = np.array(mag_phase).flatten()  # Flatten the list of tuples to a single list
            
            # Update the column with the new values
            self.data[col] = mag_phase

    def normalize_data(self):
        """
        Normalizes each observation using standard scaling (mean = 0, std = 1).
        """

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data.T).T
        return pd.DataFrame(scaled_data)
    
    def apply_pca(self):
        """
        Applies PCA to reduce the dimensionality of the dataset, considering each observation as a column.
        """

        pca = PCA(n_components=self.n_components)
        principal_components = pca.fit_transform(self.data.T).T
        return pd.DataFrame(principal_components)
    
    def get_data(self):
        """
        Get the data (204x120 matrix)
        
        @return data as a 204x120 matrix.
        """
        
        return self.data