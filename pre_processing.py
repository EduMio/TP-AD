import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import chi2
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def remove_outliers_chi_squared(df, significance_level=0.05):
    # Calculate the chi-squared critical value
    chi_squared_critical_value = chi2.ppf(1 - significance_level, df.shape[1])
    
    # Calculate the Mahalanobis distance for each row
    mean = np.mean(df, axis=0)
    cov = np.cov(df.values, rowvar=False)
    inv_covmat = np.linalg.inv(cov)
    mahalanobis_distances = df.apply(lambda row: np.dot(np.dot((row - mean), inv_covmat), (row - mean).T), axis=1)
    
    # Identify outliers
    outliers = mahalanobis_distances > chi_squared_critical_value
    
    # Remove outliers
    cleaned_df = df[~outliers]
    
    return cleaned_df

def normalize_data(df):
    scaler = MinMaxScaler(feature_range=(0, 100))
    normalized_data = scaler.fit_transform(df)
    normalized_df = pd.DataFrame(normalized_data, columns=df.columns)
    return normalized_df

# Apply the discretization function to the normalized dataset
def discretize_data_with_lower_bounds(df, bins=20):
    discretized_df = pd.DataFrame()
    for col in df.columns:
        # Get the bin edges
        bin_edges = np.linspace(df[col].min(), df[col].max(), bins + 1)
            
        # Assign each value to the lower bound of its bin
        bin_labels = bin_edges[:-1]
        discretized_col = pd.cut(df[col], bins=bin_edges, labels=bin_labels, include_lowest=True)
            
        discretized_df[col] = discretized_col.astype(float)
        
    return discretized_df

def binarize_columns(df):
    binary_dict = {}
    
    for col in df.columns:
        # Round the values to the nearest integer
        rounded_col = df[col].round().astype(int)
        unique_values = sorted(rounded_col.unique())
        
        for value in unique_values:
            binary_dict[f'{col}_{value}'] = (rounded_col == value).astype(int)
    
    binary_df = pd.DataFrame(binary_dict)
    
    return binary_df