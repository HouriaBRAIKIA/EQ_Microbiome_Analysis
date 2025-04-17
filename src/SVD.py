import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class SVD(BaseEstimator, TransformerMixin):
    """
    Class to perform Singular Value Decomposition (SVD) for dimensionality reduction.
    
    Parameters:
    -----------
    n_components : int
        The number of principal components to keep.
    """
    
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y=None):
        """
        Fit the model to the data. No fitting is needed for SVD, so it just returns self.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : ignored, default=None
            Not used, present for compatibility with sklearn's fit method.

        Returns:
        --------
        self : object
            Returns the instance itself.
        """
        
        # No need to fit for SVD
        return self

    def transform(self, X):
        """
        Apply dimensionality reduction using SVD.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data to be transformed.

        Returns:
        --------
        reduced_data : array-like of shape (n_samples, n_components)
            The data projected onto the selected components.
        """
        
        # Perform SVD dimensionality reduction
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        reduced_data = U[:, :self.n_components] * S[:self.n_components]
        return reduced_data