import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skbio.stats.ordination import cca
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr


class MultivariatesAnalysis:
    """
    A class to perform multivariate analysis, such as correlation and Canonical Correspondence Analysis (CCA).
    
    Attributes
    ----------
    metadata_data : pd.DataFrame
        Metadata containing environmental variables and EQ indices.
    otu_data_dict : Optional[pd.DataFrame]
        DataFrame containing OTU data for each sample. Default is None.
    
    Methods
    -------
    correlation_bi_env():
        Calculates the correlation between environmental variables and EQ indices.
    
    perform_cca(env_cols: list, title: str):
        Performs Canonical Correspondence Analysis (CCA) on OTU data and environmental variables.
    """
    
    def __init__(self, metadata_data, otu_data_dict=None):
        """
        Initializes the MultivariatesAnalysis object with metadata and OTU data.
        
        Parameters:
        ----------
        metadata_data : pd.DataFrame
            Metadata containing environmental variables and EQ indices.
        otu_data_dict : Optional[Dict[str, pd.DataFrame]], optional
            OTU data for each sample. Default is None.
        """
        
        self.metadata_data = metadata_data
        self.otu_data_dict = otu_data_dict

    def correlation_bi_env(self):
        """
        Calculates the Pearson correlation between environmental variables (Distance, Depth, pH) and EQ indices (AMBI, NSI, ISI, NQI1).
        
        Returns
        -------
        pd.DataFrame
            Correlation coefficients between environmental variables and EQ indices.
        pd.DataFrame
            P-values for the correlation tests.
        """
        correlations = {}
        p_values = {}
        for eq in ['AMBI', 'NSI', 'ISI', 'NQI1']:
            corr_dist, p_dist = pearsonr(self.metadata_data['Distance_cage_gps'], self.metadata_data[eq])
            corr_depth, p_depth = pearsonr(self.metadata_data['Depth'], self.metadata_data[eq])
            corr_ph, p_ph = pearsonr(self.metadata_data['pH'], self.metadata_data[eq])
            correlations[eq] = {'Distance_cage_gps': corr_dist, 'Depth': corr_depth, 'pH': corr_ph}
            p_values[eq] = {'Distance_cage_gps': p_dist, 'Depth': p_depth, 'pH': p_ph}

        # Plot heatmap of the correlations
        corr_df = pd.DataFrame(correlations)
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_df, annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap between Environmental Variables and EQ Indices")
        plt.savefig('../results/meta_corr.png')
        return pd.DataFrame(correlations), pd.DataFrame(p_values)

    def perform_cca(self, env_cols=['Distance_cage_gps', 'Depth', 'pH'], title="CCA"):
        """
        Performs Canonical Correspondence Analysis (CCA) to explore the relationship between OTU data and environmental variables.
        
        Parameters:
        ----------
        env_cols : list, optional
            List of environmental variables to be included in the CCA (default is ['Distance_cage_gps', 'Depth', 'pH']).
        title : str, optional
            Title of the plot and filename for saving (default is "CCA").
        """
        # Select only environmental variables
        env_data = self.metadata_data[env_cols].copy()

        # Run CCA (skbio expects: features in columns, samples in rows)
        ordination_result = cca(self.otu_data_dict, env_data)
        print(ordination_result)
        
        sample_scores = ordination_result.samples.iloc[:, :2]  # First two CCA axes
        biplot_scores = ordination_result.biplot_scores.iloc[:, :2]
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(sample_scores.iloc[:, 0], sample_scores.iloc[:, 1], c=list(self.metadata_data["AMBI"]), label="Samples")
        cbar = plt.colorbar(scatter)
        cbar.set_label("EQ")
        
        # Add arrows for environmental variables
        for i in range(biplot_scores.shape[0]):
            x, y = biplot_scores.iloc[i, 0], biplot_scores.iloc[i, 1]
            plt.arrow(0, 0, x * 0.1, y * 0.1, color='red', alpha=0.7, head_width=0.01, length_includes_head=True)
            plt.text(x * 0.1, y * 0.1, biplot_scores.index[i], color='red', fontsize=12)
        
        # Labels and title
        plt.xlabel("CCA Axis 1")
        plt.ylabel("CCA Axis 2")
        plt.title("Canonical Correspondence Analysis")
        plt.legend()
        plt.savefig('../results/'+title+'.png')   
        plt.show()
