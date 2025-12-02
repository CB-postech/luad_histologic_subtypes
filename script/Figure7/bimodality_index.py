#!/usr/bin/env python3
"""
Bimodality Analysis of Single-Cell RNA-seq Data
* The method orginally inspired by Wang et al. (https://doi.org/10.4137/cin.s2846)

This script analyzes bimodality in cancer cell populations using correlation-based
approaches and Gaussian mixture models. It computes bimodality indices for each
sample and performs statistical comparisons across groups.

Requirements:
    - Python 3.9+
    - scanpy
    - numpy
    - pandas
    - scipy
    - scikit-learn
    - statsmodels
    - matplotlib

Usage:
    python bimodality_analysis.py --data_dir /path/to/data --output_dir /path/to/output
"""

from __future__ import annotations

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

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


# ============================================================================
# Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Bimodality Calculation Functions
# ============================================================================

def calculate_bimodality_index(data: np.ndarray | pd.DataFrame) -> float:
    """
    Calculate bimodality index using Gaussian mixture models.
    
    The bimodality index (BI) quantifies the degree of separation between
    two subpopulations in a distribution using:
    BI = δ * sqrt(π * (1 - π))
    
    where δ is the standardized distance between means and π is the 
    mixing proportion of the minor component.
    
    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Input data for bimodality assessment.
        
    Returns
    -------
    float
        Bimodality index value.
        
    References
    ----------
    Wang et al. (2009). A mixture model with dependent observations for test reliability. 
    Journal of Data Science, 7, 105-114.
    """
    # Convert DataFrame to numpy array if needed
    if isinstance(data, pd.DataFrame):
        data = data.values.flatten()
    
    # Reshape for GMM (requires 2D array)
    data = data.reshape(-1, 1)
    
    # Fit two-component Gaussian mixture model with tied covariances
    gmm = GaussianMixture(n_components=2, covariance_type='tied', random_state=42)
    gmm.fit(data)
    
    # Extract model parameters
    means = gmm.means_.flatten()
    std_dev = np.sqrt(gmm.covariances_).flatten()[0]
    weights = gmm.weights_
    
    # Sort components by mean (ensure mean1 < mean2)
    sorted_idx = np.argsort(means)
    mean1, mean2 = means[sorted_idx]
    weight1, weight2 = weights[sorted_idx]
    
    # Calculate standardized distance between modes
    delta = abs(mean1 - mean2) / std_dev
    
    # Calculate mixing proportion of minor component
    pi_minor = min(weight1, weight2)
    
    # Compute bimodality index
    bi = delta * np.sqrt(pi_minor * (1 - pi_minor))
    
    return bi


def compute_pairwise_correlations(expr_matrix: np.ndarray) -> np.ndarray:
    """
    Compute pairwise correlation coefficients between cells.
    
    Parameters
    ----------
    expr_matrix : np.ndarray
        Expression matrix (cells × genes).
        
    Returns
    -------
    np.ndarray
        Upper triangle of correlation matrix (excluding diagonal).
    """
    # Compute cell-cell correlation matrix
    corr_matrix = np.corrcoef(expr_matrix)
    
    # Extract upper triangle (excluding diagonal)
    upper_tri_idx = np.triu_indices(corr_matrix.shape[0], k=1)
    upper_tri_values = corr_matrix[upper_tri_idx]
    
    return upper_tri_values


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

def load_data(data_dir: Path, 
              sample_groups: Dict[str, str]) -> ad.AnnData:
    """
    Load and preprocess single-cell RNA-seq data.
    
    Parameters
    ----------
    data_dir : Path
        Directory containing 10X format data (matrix.mtx, barcodes.tsv, genes.tsv).
    sample_groups : Dict[str, str]
        Mapping of sample IDs to group labels.
        
    Returns
    -------
    ad.AnnData
        Annotated data object with metadata.
    """
    logger.info(f"Loading data from {data_dir}")
    
    # Load 10X format data
    adata = sc.read_10x_mtx(data_dir, var_names='gene_symbols', cache=True)
    
    # Load metadata
    metadata_path = data_dir / "metadata.csv"
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path, index_col=0)
        adata.obs = metadata.loc[adata.obs_names, :].copy()
    
    # Rename columns for standardization
    if 'specimenID' in adata.obs.columns:
        adata.obs = adata.obs.rename(columns={'specimenID': 'sample_id'})
    if 'histogroup' in adata.obs.columns:
        adata.obs = adata.obs.rename(columns={'histogroup': 'group'})
    
    # Map samples to groups
    if 'sample_id' in adata.obs.columns:
        adata.obs['group'] = adata.obs['sample_id'].map(sample_groups)
    
    logger.info(f"Loaded {adata.shape[0]} cells × {adata.shape[1]} genes")
    logger.info(f"Group distribution:\n{adata.obs['group'].value_counts()}")
    
    return adata


def identify_highly_variable_genes(adata: ad.AnnData, 
                                   n_top_genes: int = 5000) -> ad.AnnData:
    """
    Identify highly variable genes for downstream analysis.
    
    Parameters
    ----------
    adata : ad.AnnData
        Annotated data object.
    n_top_genes : int
        Number of top variable genes to select.
        
    Returns
    -------
    ad.AnnData
        AnnData object with HVG annotation in .var['highly_variable'].
    """
    logger.info(f"Identifying top {n_top_genes} highly variable genes")
    
    sc.pp.highly_variable_genes(
        adata, 
        n_top_genes=n_top_genes, 
        flavor='seurat_v3', 
        subset=False
    )
    
    n_hvg = adata.var['highly_variable'].sum()
    logger.info(f"Identified {n_hvg} highly variable genes")
    
    return adata


# ============================================================================
# Bimodality Analysis
# ============================================================================

def analyze_sample_bimodality(adata: ad.AnnData, 
                              sample_id: str,
                              hvg_genes: pd.Index) -> Tuple[str, float]:
    """
    Compute bimodality index for a single sample.
    
    Parameters
    ----------
    adata : ad.AnnData
        Full annotated data object.
    sample_id : str
        Sample identifier to analyze.
    hvg_genes : pd.Index
        Highly variable gene names.
        
    Returns
    -------
    Tuple[str, float]
        Sample ID and bimodality index.
    """
    logger.info(f"Processing sample: {sample_id}")
    
    # Subset to sample
    adata_sample = adata[adata.obs['sample_id'] == sample_id].copy()
    
    # Normalize and log-transform
    sc.pp.normalize_total(adata_sample, target_sum=1e4)
    sc.pp.log1p(adata_sample)
    
    # Subset to highly variable genes
    adata_sample = adata_sample[:, hvg_genes].copy()
    
    # Convert to dense matrix if sparse
    if sp.issparse(adata_sample.X):
        expr_matrix = adata_sample.X.toarray()
    else:
        expr_matrix = adata_sample.X
    
    # Calculate pairwise correlations
    correlations = compute_pairwise_correlations(expr_matrix)
    
    # Compute bimodality index
    corr_df = pd.DataFrame({'correlation': correlations})
    bi = calculate_bimodality_index(corr_df)
    
    logger.info(f"  Bimodality index: {bi:.4f}")
    
    return sample_id, bi


def run_bimodality_analysis(adata: ad.AnnData, 
                            sample_col: str = 'sample_id') -> pd.DataFrame:
    """
    Perform bimodality analysis across all samples.
    
    Parameters
    ----------
    adata : ad.AnnData
        Annotated data object with HVG annotation.
    sample_col : str
        Column name containing sample identifiers.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with bimodality indices for each sample.
    """
    # Get highly variable genes
    hvg_genes = adata.var_names[adata.var['highly_variable']]
    
    # Analyze each sample
    samples = adata.obs[sample_col].unique()
    results = []
    
    for sample_id in samples:
        sample_id_str, bi = analyze_sample_bimodality(adata, sample_id, hvg_genes)
        results.append({'sample': sample_id_str, 'bimodality_index': bi})
    
    # Create results DataFrame
    df_results = pd.DataFrame(results)
    
    # Add group information
    sample_to_group = adata.obs.groupby(sample_col)['group'].first().to_dict()
    df_results['group'] = df_results['sample'].map(sample_to_group)
    
    return df_results


# ============================================================================
# Statistical Analysis
# ============================================================================

def perform_statistical_tests(df: pd.DataFrame) -> None:
    """
    Perform ANOVA and post-hoc tests on bimodality indices.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'bimodality_index' and 'group' columns.
    """
    logger.info("\n" + "="*60)
    logger.info("Statistical Analysis")
    logger.info("="*60)
    
    # One-way ANOVA
    logger.info("\nOne-way ANOVA:")
    model = ols('bimodality_index ~ C(group)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)
    
    # Tukey's HSD post-hoc test
    logger.info("\nTukey's HSD Post-hoc Test:")
    tukey = pairwise_tukeyhsd(
        endog=df['bimodality_index'], 
        groups=df['group'], 
        alpha=0.05
    )
    print(tukey)


# ============================================================================
# Visualization
# ============================================================================

def plot_bimodality_boxplot(df: pd.DataFrame, 
                            output_path: Path | None = None) -> None:
    """
    Create boxplot visualization of bimodality indices by group.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'group' and 'bimodality_index' columns.
    output_path : Path, optional
        Path to save figure. If None, displays plot.
    """
    logger.info("Creating bimodality boxplot")
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for boxplot
    groups = df['group'].unique()
    boxplot_data = [df[df['group'] == g]['bimodality_index'].values for g in groups]
    
    # Create boxplot
    bp = ax.boxplot(boxplot_data, patch_artist=True, notch=False, widths=0.6)
    
    # Color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
    
    # Customize boxes
    for patch, color in zip(bp['boxes'], colors[:len(groups)]):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(1)
    
    # Overlay individual points
    for i, group in enumerate(groups):
        group_data = df[df['group'] == group]
        x_coords = np.full(len(group_data), i + 1)
        y_coords = group_data['bimodality_index'].values
        ax.scatter(x_coords, y_coords, color=colors[i], alpha=0.7, s=50, zorder=3)
    
    # Customize plot
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(groups, rotation=45, ha='right')
    ax.set_xlabel('Group', fontsize=12)
    ax.set_ylabel('Bimodality Index', fontsize=12)
    ax.set_title('Bimodality Index by Group', fontsize=14, fontweight='bold')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# Main Workflow
# ============================================================================

def main():
    """Main analysis workflow."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Bimodality analysis of single-cell RNA-seq data'
    )
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default='/path/to/data',
        help='Directory containing 10X format data'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='/path/to/output',
        help='Output directory for results'
    )
    parser.add_argument(
        '--n_hvg', 
        type=int, 
        default=5000,
        help='Number of highly variable genes to use'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define sample groupings
    # NOTE: Update this mapping based on your experimental groups
    sample_groups = {
        'Sample_01': 'Group_A',
        'Sample_02': 'Group_B',
        'Sample_03': 'Group_B',
        'Sample_04': 'Group_C',
        'Sample_05': 'Group_C',
        'Sample_06': 'Group_C',
        'Sample_07': 'Group_B',
        'Sample_08': 'Group_C',
        'Sample_09': 'Group_A',
        'Sample_10': 'Group_B',
        'Sample_11': 'Group_D',
        'Sample_12': 'Group_C',
        'Sample_13': 'Group_D',
        'Sample_14': 'Group_D',
        'Sample_15': 'Group_A',
        'Sample_16': 'Group_D',
        'Sample_17': 'Group_D',
        'Sample_18': 'Group_D',
    }
    
    # Load data
    adata = load_data(data_dir, sample_groups)
    
    # Identify highly variable genes
    adata = identify_highly_variable_genes(adata, n_top_genes=args.n_hvg)
    
    # Run bimodality analysis
    logger.info("\n" + "="*60)
    logger.info("Running Bimodality Analysis")
    logger.info("="*60)
    df_bimodality = run_bimodality_analysis(adata, sample_col='sample_id')
    
    # Display results
    logger.info("\nBimodality Results:")
    print(df_bimodality.to_string(index=False))
    
    # Save results
    results_path = output_dir / 'bimodality_results.csv'
    df_bimodality.to_csv(results_path, index=False)
    logger.info(f"\nSaved results to {results_path}")
    
    # Statistical analysis
    perform_statistical_tests(df_bimodality)
    
    # Create visualization
    plot_path = output_dir / 'bimodality_boxplot.png'
    plot_bimodality_boxplot(df_bimodality, output_path=plot_path)
    
    logger.info("\nAnalysis complete!")


if __name__ == '__main__':
    main()
