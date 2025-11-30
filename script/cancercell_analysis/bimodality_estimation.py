### The method orginally inspired by Wang et al

#!/usr/bin/env python3
"""
Cancer Cell Bimodality Analysis
================================
Analyzes clonal heterogeneity in cancer cells using cell-cell correlation bimodality.

This script:
1. Loads single-cell RNA-seq data of cancer cells
2. Identifies highly variable genes
3. Calculates cell-cell correlation matrices per sample
4. Computes bimodality index using Gaussian Mixture Models
5. Performs statistical analysis (ANOVA + Tukey HSD)
6. Generates visualization
"""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

import anndata as ad
import scanpy as sc
from sklearn.mixture import GaussianMixture
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Data paths
DATA_DIR = Path("data/fig4_cancer_cells")
METADATA_FILE = DATA_DIR / "metadata.csv"

# Analysis parameters
N_HVG = 5000  # Number of highly variable genes
TARGET_SUM = 1e4  # Normalization target
ALPHA_LEVEL = 0.05  # Significance level for statistical tests

# Histologic subtype mapping
HISTOLOGIC_GROUPS = {
    "Sample01": "MP", "Sample02": "Solid", "Sample03": "Solid",
    "Sample04": "AP+Solid", "Sample05": "AP+Solid", "Sample06": "AP+Solid",
    "Sample07": "Solid", "Sample08": "AP+Solid", "Sample09": "MP",
    "Sample11": "Solid", "Sample12": "AP", "Sample13": "AP+Solid",
    "Sample14": "AP", "Sample16": "AP", "Sample18": "MP",
    "Sample19": "AP", "Sample20": "AP", "Sample22": "AP"
}

# Group classification for analysis
GROUP_MAPPING = {
    'Sample01': 'G4_MP', 'Sample02': 'G3_Solid', 'Sample03': 'G3_Solid',
    'Sample04': 'G2_AP+Solid', 'Sample05': 'G2_AP+Solid', 'Sample06': 'G2_AP+Solid',
    'Sample07': 'G3_Solid', 'Sample08': 'G2_AP+Solid', 'Sample09': 'G4_MP',
    'Sample11': 'G3_Solid', 'Sample12': 'G1_AP', 'Sample13': 'G2_AP+Solid',
    'Sample14': 'G1_AP', 'Sample16': 'G1_AP', 'Sample18': 'G4_MP',
    'Sample19': 'G1_AP', 'Sample20': 'G1_AP', 'Sample22': 'G1_AP'
}

# ══════════════════════════════════════════════════════════════════════════════
# Core Functions
# ══════════════════════════════════════════════════════════════════════════════

def calc_bimodality_index(data: np.ndarray | pd.DataFrame) -> float:
    """
    Calculate bimodality index using Gaussian Mixture Model.
    
    The bimodality index measures the separation between two distributions:
    BI = δ * sqrt(π * (1 - π))
    where δ is the standardized distance between means and π is the minimum weight.
    
    Parameters
    ----------
    data : array-like
        Correlation coefficients or other continuous data
        
    Returns
    -------
    float
        Bimodality index (higher = more bimodal)
    """
    # Convert to numpy array
    if isinstance(data, pd.DataFrame):
        data = data.values.flatten()
    
    # Reshape for GMM
    data = data.reshape(-1, 1)
    
    # Fit mixture of two Gaussians
    gmm = GaussianMixture(n_components=2, covariance_type='tied', random_state=42)
    gmm.fit(data)
    
    # Extract parameters
    means = gmm.means_.flatten()
    std_dev = np.sqrt(gmm.covariances_).flatten()[0]
    weights = gmm.weights_
    
    # Sort to ensure mean1 < mean2
    if means[0] > means[1]:
        means = means[::-1]
        weights = weights[::-1]
    
    mean1, mean2 = means
    weight1, weight2 = weights
    
    # Calculate standardized distance and bimodality index
    delta = abs(mean1 - mean2) / std_dev
    pi = min(weight1, weight2)
    bi = delta * np.sqrt(pi * (1 - pi))
    
    return bi


def load_data(data_dir: Path, metadata_path: Path) -> ad.AnnData:
    """
    Load 10X format data and metadata.
    
    Parameters
    ----------
    data_dir : Path
        Directory containing matrix.mtx, barcodes.tsv, genes.tsv
    metadata_path : Path
        Path to metadata CSV file
        
    Returns
    -------
    AnnData
        Annotated data object with metadata
    """
    logging.info("Loading 10X format data...")
    adata = sc.read_10x_mtx(data_dir, var_names='gene_symbols', cache=True)
    
    logging.info("Loading metadata...")
    metadata = pd.read_csv(metadata_path, index_col=0)
    
    # Align and add metadata
    adata.obs = metadata.loc[adata.obs_names, :].copy()
    
    logging.info(f"Loaded {adata.shape[0]} cells × {adata.shape[1]} genes")
    
    return adata


def prepare_data(adata: ad.AnnData) -> ad.AnnData:
    """
    Prepare data by updating histologic subtypes.
    
    Parameters
    ----------
    adata : AnnData
        Input annotated data
        
    Returns
    -------
    AnnData
        Processed annotated data with updated subtypes
    """
    # Update histologic subtypes with ground truth
    adata.obs['hist_subtype'] = adata.obs['specimenID'].map(HISTOLOGIC_GROUPS)
    
    logging.info(f"Subtype distribution:\n{adata.obs['hist_subtype'].value_counts()}")
    logging.info(f"Total samples: {adata.obs['specimenID'].nunique()}")
    
    return adata


def identify_hvg(adata: ad.AnnData, n_top_genes: int = 5000) -> None:
    """
    Identify highly variable genes.
    
    Parameters
    ----------
    adata : AnnData
        Input data (modified in place)
    n_top_genes : int
        Number of top variable genes to select
    """
    logging.info(f"Identifying top {n_top_genes} highly variable genes...")
    sc.pp.highly_variable_genes(
        adata, 
        n_top_genes=n_top_genes, 
        flavor='seurat_v3', 
        subset=False
    )
    n_hvg = adata.var['highly_variable'].sum()
    logging.info(f"Found {n_hvg} highly variable genes")


def calculate_sample_bimodality(
    adata: ad.AnnData, 
    sample_id: str,
    target_sum: float = 1e4
) -> float:
    """
    Calculate bimodality index for a single sample.
    
    Parameters
    ----------
    adata : AnnData
        Full dataset with HVG annotations
    sample_id : str
        Sample identifier
    target_sum : float
        Normalization target sum
        
    Returns
    -------
    float
        Bimodality index
    """
    # Subset to sample
    adata_subset = adata[adata.obs["specimenID"] == sample_id].copy()
    
    # Normalize and log-transform
    sc.pp.normalize_total(adata_subset, target_sum=target_sum)
    sc.pp.log1p(adata_subset)
    
    # Subset to HVG
    hvg_genes = adata.var_names[adata.var['highly_variable']]
    adata_subset = adata_subset[:, hvg_genes].copy()
    
    # Get expression matrix
    if sp.issparse(adata_subset.X):
        hvg_matrix = adata_subset.X.toarray()
    else:
        hvg_matrix = adata_subset.X
    
    # Calculate cell-cell correlation matrix
    corr_matrix = np.corrcoef(hvg_matrix)
    
    # Extract upper triangle (exclude diagonal)
    upper_tri_indices = np.triu_indices(corr_matrix.shape[0], k=1)
    correlations = corr_matrix[upper_tri_indices]
    
    # Calculate bimodality index
    bimodality_index = calc_bimodality_index(correlations)
    
    return bimodality_index


def calculate_bimodality_all_samples(adata: ad.AnnData) -> pd.DataFrame:
    """
    Calculate bimodality indices for all samples.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data with HVG identified
        
    Returns
    -------
    DataFrame
        Bimodality indices with sample and group information
    """
    logging.info("Calculating bimodality indices for all samples...")
    
    results = []
    sample_list = adata.obs["specimenID"].unique()
    
    for sample_id in sample_list:
        logging.info(f"Processing {sample_id}...")
        bi = calculate_sample_bimodality(adata, sample_id, TARGET_SUM)
        logging.info(f"  Bimodality index: {bi:.3f}")
        results.append({'sample': sample_id, 'bimodality_index': bi})
    
    # Create dataframe
    bimodality_df = pd.DataFrame(results)
    bimodality_df['group'] = bimodality_df['sample'].map(GROUP_MAPPING)
    
    logging.info(f"\nCompleted {len(results)} samples")
    
    return bimodality_df


def perform_statistical_analysis(bimodality_df: pd.DataFrame, alpha: float = 0.05) -> None:
    """
    Perform ANOVA and Tukey HSD post-hoc test.
    
    Parameters
    ----------
    bimodality_df : DataFrame
        Bimodality indices with group information
    alpha : float
        Significance level
    """
    logging.info("\n" + "="*80)
    logging.info("STATISTICAL ANALYSIS")
    logging.info("="*80)
    
    # ANOVA
    model = ols('bimodality_index ~ C(group)', data=bimodality_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    print("\nANOVA Table:")
    print(anova_table)
    
    # Tukey HSD
    tukey = pairwise_tukeyhsd(
        endog=bimodality_df['bimodality_index'], 
        groups=bimodality_df['group'], 
        alpha=alpha
    )
    
    print("\nTukey's HSD Test Results:")
    print(tukey)


def create_boxplot(bimodality_df: pd.DataFrame, output_path: str = None) -> None:
    """
    Create boxplot with individual data points.
    
    Parameters
    ----------
    bimodality_df : DataFrame
        Bimodality indices with group information
    output_path : str, optional
        Path to save figure
    """
    logging.info("Creating visualization...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for boxplot
    groups = bimodality_df['group'].unique()
    boxplot_data = [
        bimodality_df[bimodality_df['group'] == group]['bimodality_index'] 
        for group in groups
    ]
    
    # Create boxplot
    bp = ax.boxplot(boxplot_data, patch_artist=True, notch=False, widths=0.6)
    
    # Color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(1)
    
    # Add individual data points
    for i, group in enumerate(groups):
        group_data = bimodality_df[bimodality_df['group'] == group]
        x_coords = [i + 1] * len(group_data)
        y_coords = group_data['bimodality_index']
        ax.scatter(x_coords, y_coords, color=colors[i], alpha=0.8, s=30, zorder=3)
    
    # Styling
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(groups, rotation=45, ha='right')
    ax.set_xlabel('Group', fontsize=12)
    ax.set_ylabel('Bimodality Index', fontsize=12)
    ax.set_title('Cancer Cell Clonal Heterogeneity by Histologic Subtype', 
                 fontsize=14, fontweight='bold')
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"Figure saved to {output_path}")
    
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Run the complete bimodality analysis pipeline."""
    
    logging.info("="*80)
    logging.info("CANCER CELL BIMODALITY ANALYSIS")
    logging.info("="*80)
    
    # Load data
    adata = load_data(DATA_DIR, METADATA_FILE)
    
    # Prepare data
    adata = prepare_data(adata)
    
    # Identify highly variable genes
    identify_hvg(adata, n_top_genes=N_HVG)
    
    # Calculate bimodality indices
    bimodality_df = calculate_bimodality_all_samples(adata)
    
    # Display results
    print("\n" + "="*80)
    print("BIMODALITY RESULTS")
    print("="*80)
    print(bimodality_df)
    
    # Statistical analysis
    perform_statistical_analysis(bimodality_df, alpha=ALPHA_LEVEL)
    
    # Visualization
    create_boxplot(bimodality_df, output_path="bimodality_boxplot.png")
    
    # Save results
    output_file = "bimodality_results.csv"
    bimodality_df.to_csv(output_file, index=False)
    logging.info(f"\nResults saved to {output_file}")
    
    logging.info("\n" + "="*80)
    logging.info("ANALYSIS COMPLETE")
    logging.info("="*80)


if __name__ == "__main__":
    main()
