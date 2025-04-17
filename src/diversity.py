import pandas as pd
from skbio.diversity.alpha import shannon, simpson, pielou_e, observed_otus, sobs
import matplotlib.pyplot as plt

class DiversityAnalysis:
    """
    A class to perform alpha diversity analysis on ecological data.

    Attributes
    ----------
    data : pd.DataFrame
        A DataFrame where rows represent samples and columns represent OTUs.
    
    Methods
    -------
    calculate_alpha_diversity() -> pd.DataFrame
        Calculates alpha diversity indices for each sample, including:
        - Species Richness
        - Shannon Index
        - Simpson Index
        - Pielou's Evenness
    """

    def __init__(self, data):
        """
        Initializes the DiversityAnalysis object.

        Parameters
        ----------
        data : pd.DataFrame
            A DataFrame with OTU counts where rows are samples and columns are OTUs.
        """
        
        self.data = data

    def calculate_alpha_diversity(self):
        """
        Calculate alpha diversity indices for each sample.
        
        The following indices are calculated:
        - Species Richness: The number of species present in a sample.
        - Shannon Index: A measure of diversity that accounts for both abundance and evenness.
        - Simpson Index: A measure of dominance, where a higher value indicates lower diversity.
        - Pielou's Evenness: A measure of evenness, which normalizes the Shannon index.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the calculated diversity indices for each sample.
        """
        
        # Calcul de l'indice de diversité alpha
        div_index = pd.DataFrame()

        div_index['Species_Richness'] = (self.data > 0).sum(axis=1)
        div_index['Shannon_Index'] = self.data.apply(lambda x: shannon(x.values), axis=1)
        div_index['Simpson_Index'] = self.data.apply(lambda x: simpson(x.values), axis=1)
        div_index['Pielou_Evenness'] = self.data.apply(lambda x: pielou_e(x.values), axis=1)

        return div_index

