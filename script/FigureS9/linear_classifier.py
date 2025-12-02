#!/usr/bin/env python3
"""
Linear classifier for histological subtype prediction based on scRNA-seq of cancer cells

"""

from __future__ import annotations

import os
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

import anndata as ad
import scanpy as sc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


# ============================================================================
# Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Plotting configuration
sns.set_theme(style='whitegrid', font='DejaVu Sans')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

RANDOM_STATE = 42


# ============================================================================
# Utility Functions
# ============================================================================

def check_counts_integer(adata: ad.AnnData) -> None:
    """
    Verify that count data contains integer values.
    
    Parameters
    ----------
    adata : ad.AnnData
        Annotated data object.
    """
    X = adata.X
    if sp.issparse(X):
        X_is_integer = X.data.size == 0 or np.all(np.equal(X.data, np.floor(X.data)))
    else:
        X_is_integer = np.all(np.equal(X, np.floor(X)))
    logger.info(f'adata.X is integer: {X_is_integer}')
    
    if 'counts' in adata.layers:
        C = adata.layers['counts']
        if sp.issparse(C):
            counts_is_integer = C.data.size == 0 or np.all(np.equal(C.data, np.floor(C.data)))
        else:
            counts_is_integer = np.all(np.equal(C, np.floor(C)))
        logger.info(f"adata.layers['counts'] is integer: {counts_is_integer}")


def pseudobulk_sum(adata: ad.AnnData, 
                   group_key: str = 'sample_id', 
                   layer: str = 'counts') -> pd.DataFrame:
    """
    Aggregate single-cell counts to pseudobulk by summing across cells per group.
    
    Parameters
    ----------
    adata : ad.AnnData
        Annotated data object containing single-cell data.
    group_key : str
        Column in adata.obs defining sample/patient groups.
    layer : str
        Layer name containing count data. If not found, uses adata.X.
        
    Returns
    -------
    pd.DataFrame
        Pseudobulk count matrix (samples × genes).
    """
    assert group_key in adata.obs, f'Missing {group_key} in obs'
    
    X = adata.layers.get(layer, adata.X)
    genes = adata.var_names.astype(str)
    ids = adata.obs[group_key].astype(str).values
    uniq = np.unique(ids)
    
    bulk_rows = []
    bulk_index = []
    
    for group_id in uniq:
        mask = ids == group_id
        if sp.issparse(X):
            s = np.asarray(X[mask].sum(axis=0)).ravel()
        else:
            s = X[mask].sum(axis=0).ravel()
        bulk_rows.append(s)
        bulk_index.append(group_id)
    
    df = pd.DataFrame(
        np.vstack(bulk_rows),
        index=pd.Index(bulk_index, name=group_key),
        columns=genes
    )
    return df


def log_cpm(counts: pd.DataFrame, target_sum: float = 1e6) -> pd.DataFrame:
    """
    Transform count data to log-CPM (counts per million).
    
    Parameters
    ----------
    counts : pd.DataFrame
        Count matrix (samples × genes).
    target_sum : float
        Normalization target (default: 1 million).
        
    Returns
    -------
    pd.DataFrame
        Log-transformed CPM values.
    """
    lib_size = counts.sum(axis=1).replace(0, np.nan)
    cpm = counts.div(lib_size, axis=0) * target_sum
    return np.log1p(cpm).fillna(0.0)


def select_pc1_genes(X_train: np.ndarray, 
                     gene_names: np.ndarray, 
                     topk: int) -> pd.Index:
    """
    Select top-K genes based on PC1 loadings from training data.
    
    This function performs PCA-based feature selection without data leakage
    by operating only on training data.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training expression matrix (samples × genes).
    gene_names : np.ndarray
        Gene names corresponding to columns of X_train.
    topk : int
        Number of top genes to select.
        
    Returns
    -------
    pd.Index
        Selected gene names.
    """
    # Standardize training data
    scaler = StandardScaler(with_mean=True, with_std=True).fit(X_train)
    X_scaled = scaler.transform(X_train)
    
    # Fit PCA (limited by min of samples and genes)
    n_components = int(min(50, X_scaled.shape[0], X_scaled.shape[1])) or 1
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE).fit(X_scaled)
    
    # Extract PC1 loadings
    pc1_loadings = pd.Series(pca.components_[0], index=gene_names)
    
    # Select top-K by absolute loading
    return pc1_loadings.abs().nlargest(topk).index


def evaluate_predictions(y_true: np.ndarray, 
                        y_pred: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate classification metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
        
    Returns
    -------
    Tuple[float, float, float]
        Accuracy, balanced accuracy, and macro F1 score.
    """
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    return acc, bal_acc, f1_macro


def plot_confusion(y_true: np.ndarray, 
                   y_pred: np.ndarray, 
                   title: str,
                   save_path: Optional[Path] = None) -> None:
    """
    Plot confusion matrix heatmap.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    title : str
        Plot title.
    save_path : Path, optional
        Path to save figure.
    """
    labels = sorted(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# Data Loading
# ============================================================================

def load_single_cell_data(data_dir: Path,
                          metadata_path: Path,
                          sample_mapping: Dict[str, str]) -> ad.AnnData:
    """
    Load and prepare single-cell RNA-seq data.
    
    Parameters
    ----------
    data_dir : Path
        Directory containing 10X format data.
    metadata_path : Path
        Path to metadata CSV file.
    sample_mapping : Dict[str, str]
        Mapping of sample IDs to subtype labels.
        
    Returns
    -------
    ad.AnnData
        Prepared annotated data object.
    """
    logger.info(f"Loading data from {data_dir}")
    
    # Load 10X data
    adata = sc.read_10x_mtx(data_dir)
    
    # Load and merge metadata
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path, index_col=0)
        adata.obs = adata.obs.join(metadata.reindex(adata.obs_names))
        
    # Store raw counts
    adata.layers['counts'] = adata.X.copy()
    adata.raw = adata.copy()
    
    # Add subtype labels
    if 'sample_id' in adata.obs.columns:
        adata.obs['subtype'] = adata.obs['sample_id'].map(sample_mapping)
    
    check_counts_integer(adata)
    logger.info(f'Loaded {adata.n_obs} cells × {adata.n_vars} genes')
    
    return adata


# ============================================================================
# PCA Visualization
# ============================================================================

def visualize_pca(adata: ad.AnnData,
                  pseudobulk_data: pd.DataFrame,
                  metadata: pd.DataFrame,
                  n_hvg: int,
                  title: str,
                  colors: Dict[str, str],
                  save_path: Optional[Path] = None) -> None:
    """
    Create PCA visualization of pseudobulk data.
    
    Parameters
    ----------
    adata : ad.AnnData
        Single-cell data for HVG selection.
    pseudobulk_data : pd.DataFrame
        Pseudobulk expression matrix.
    metadata : pd.DataFrame
        Sample metadata with subtype labels.
    n_hvg : int
        Number of highly variable genes to select.
    title : str
        Plot title.
    colors : Dict[str, str]
        Color mapping for subtypes.
    save_path : Path, optional
        Path to save figure.
    """
    logger.info(f"Creating PCA visualization with {n_hvg} HVGs")
    
    # Select HVGs
    adata_hvg = adata.copy()
    sc.pp.highly_variable_genes(
        adata_hvg,
        n_top_genes=n_hvg,
        flavor='seurat_v3',
        subset=False,
        batch_key='sample_id'
    )
    hvg_genes = adata_hvg.var_names[adata_hvg.var['highly_variable']].astype(str)
    
    # Filter to HVGs and standardize
    X_pca = pseudobulk_data[hvg_genes].values
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X_pca)
    
    # Fit PCA
    pca = PCA(n_components=min(10, X_scaled.shape[0]), random_state=RANDOM_STATE)
    pca_coords = pca.fit_transform(X_scaled)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    pca_df = pd.DataFrame({
        'PC1': pca_coords[:, 0],
        'PC2': pca_coords[:, 1],
        'Subtype': metadata['subtype'].values,
        'Sample': metadata.index
    })
    
    for subtype, color in colors.items():
        mask = pca_df['Subtype'] == subtype
        ax.scatter(pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'],
                   c=color, label=subtype, s=120, edgecolors='black',
                   linewidths=0.5, alpha=0.8)
        
        # Add sample labels
        for _, row in pca_df[mask].iterrows():
            ax.annotate(row['Sample'], (row['PC1'], row['PC2']),
                        fontsize=9, ha='center', va='bottom', alpha=0.7)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(title='Subtype', loc='best', fontsize=11)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved PCA plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# Cross-Validation with Leakage-Free Feature Selection
# ============================================================================

def run_loocv_classifier(adata: ad.AnnData,
                        pseudobulk_counts: pd.DataFrame,
                        metadata: pd.DataFrame,
                        topk: int,
                        n_hvg: int = 2000,
                        classifier_name: str = "classifier") -> Dict:
    """
    Perform leave-one-out cross-validation with leakage-free feature selection.
    
    Key approach to prevent data leakage:
    1. Hold out test sample
    2. Select HVGs from training samples' single cells only
    3. Create pseudobulk from training samples using training HVGs
    4. Select PC1 genes from training pseudobulk
    5. Apply same feature selection to test sample
    6. Train classifier and predict
    
    Parameters
    ----------
    adata : ad.AnnData
        Single-cell data for HVG selection.
    pseudobulk_counts : pd.DataFrame
        Full pseudobulk count matrix (all genes).
    metadata : pd.DataFrame
        Sample metadata with subtype labels.
    topk : int
        Number of PC1 genes to select.
    n_hvg : int
        Number of highly variable genes to select per fold.
    classifier_name : str
        Name for logging purposes.
        
    Returns
    -------
    Dict
        Dictionary containing predictions, true labels, and metrics.
    """
    logger.info(f"Starting LOOCV for {classifier_name}")
    logger.info(f"  Samples: {len(metadata)}")
    logger.info(f"  HVGs per fold: {n_hvg}")
    logger.info(f"  PC1 genes: {topk}")
    
    y_all = metadata['subtype'].astype(str).values
    sample_ids = metadata.index.values
    
    loo = LeaveOneOut()
    y_true_list, y_pred_list, y_proba_list = [], [], []
    
    for fold, (train_idx, test_idx) in enumerate(loo.split(sample_ids), 1):
        # Get train/test sample IDs
        train_samples = sample_ids[train_idx]
        test_sample = sample_ids[test_idx][0]
        
        # === STEP 1: Select HVGs on training single cells only ===
        adata_train = adata[adata.obs['sample_id'].isin(train_samples)].copy()
        sc.pp.highly_variable_genes(
            adata_train,
            n_top_genes=n_hvg,
            flavor='seurat_v3',
            subset=False,
            batch_key='sample_id'
        )
        hvg_train = adata_train.var_names[adata_train.var['highly_variable']].astype(str)
        
        # === STEP 2: Create training pseudobulk with training HVGs ===
        pb_train_counts = pseudobulk_sum(adata_train, 'sample_id', 'counts')
        pb_train_log = log_cpm(pb_train_counts)
        pb_train_hvg = pb_train_log.loc[:, hvg_train]
        
        # === STEP 3: Select PC1 genes from training pseudobulk ===
        selected_genes = select_pc1_genes(pb_train_hvg.values, hvg_train, topk)
        
        # === STEP 4: Create test pseudobulk ===
        adata_test = adata[adata.obs['sample_id'] == test_sample].copy()
        pb_test_counts = pseudobulk_sum(adata_test, 'sample_id', 'counts')
        pb_test_log = log_cpm(pb_test_counts)
        
        # === STEP 5: Extract features ===
        X_train = pb_train_hvg[selected_genes].values
        X_test = pb_test_log[selected_genes].values
        
        # === STEP 6: Standardize (fit on train only) ===
        scaler = StandardScaler(with_mean=True, with_std=True).fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # === STEP 7: Train classifier ===
        y_train = metadata.loc[train_samples, 'subtype'].values
        clf = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='lbfgs',
            max_iter=2000,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ).fit(X_train_scaled, y_train)
        
        # === STEP 8: Predict ===
        y_test = metadata.loc[test_sample, 'subtype']
        y_pred = clf.predict(X_test_scaled)[0]
        y_proba = clf.predict_proba(X_test_scaled)[0]
        
        y_true_list.append(y_test)
        y_pred_list.append(y_pred)
        y_proba_list.append(y_proba)
        
        # Log progress
        if fold % 5 == 0 or fold == len(sample_ids):
            acc, bal, f1m = evaluate_predictions(
                np.array(y_true_list),
                np.array(y_pred_list)
            )
            logger.info(f"  Fold {fold:02d}/{len(sample_ids)} | "
                       f"Acc: {acc:.3f}, Bal: {bal:.3f}, F1: {f1m:.3f}")
    
    # Calculate final metrics
    y_true_array = np.array(y_true_list)
    y_pred_array = np.array(y_pred_list)
    y_proba_array = np.array(y_proba_list)
    
    acc, bal_acc, f1_macro = evaluate_predictions(y_true_array, y_pred_array)
    
    results = {
        'y_true': y_true_array,
        'y_pred': y_pred_array,
        'y_proba': y_proba_array,
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'f1_macro': f1_macro
    }
    
    # Add binary-specific metrics if applicable
    if len(np.unique(y_true_array)) == 2:
        classes = sorted(np.unique(y_true_array))
        results['f1_class1'] = f1_score(y_true_array, y_pred_array, pos_label=classes[1])
        results['auc'] = roc_auc_score(
            (y_true_array == classes[1]).astype(int),
            y_proba_array[:, 1]
        )
    
    logger.info(f"\n{'='*70}")
    logger.info(f"LOOCV Results: {classifier_name} (top-{topk})")
    logger.info(f"{'='*70}")
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Balanced Accuracy: {bal_acc:.4f}")
    logger.info(f"F1 (Macro): {f1_macro:.4f}")
    if 'auc' in results:
        logger.info(f"ROC-AUC: {results['auc']:.4f}")
    logger.info(f"{'='*70}\n")
    
    return results


# ============================================================================
# Final Model Training
# ============================================================================

def train_final_model(adata: ad.AnnData,
                     pseudobulk_log: pd.DataFrame,
                     metadata: pd.DataFrame,
                     topk: int,
                     n_hvg: int = 2000) -> Dict:
    """
    Train final deployment model on entire dataset.
    
    Parameters
    ----------
    adata : ad.AnnData
        Single-cell data for HVG selection.
    pseudobulk_log : pd.DataFrame
        Log-transformed pseudobulk expression.
    metadata : pd.DataFrame
        Sample metadata.
    topk : int
        Number of PC1 genes to select.
    n_hvg : int
        Number of HVGs to select.
        
    Returns
    -------
    Dict
        Model bundle containing classifier, scaler, and selected genes.
    """
    logger.info(f"Training final model with top-{topk} genes")
    
    # Global HVG selection
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_hvg,
        flavor='seurat_v3',
        subset=False,
        batch_key='sample_id'
    )
    hvg_genes = adata.var_names[adata.var['highly_variable']].astype(str)
    
    # Filter to HVGs
    X_hvg = pseudobulk_log.loc[:, hvg_genes]
    
    # PC1 gene selection
    scaler_pca = StandardScaler(with_mean=True, with_std=True).fit(X_hvg.values)
    X_pca = scaler_pca.transform(X_hvg.values)
    n_comps = int(min(50, X_pca.shape[0], X_pca.shape[1])) or 1
    pca = PCA(n_components=n_comps, random_state=RANDOM_STATE).fit(X_pca)
    pc1_loadings = pd.Series(pca.components_[0], index=hvg_genes)
    selected_genes = pc1_loadings.abs().nlargest(topk).index
    
    logger.info(f"  Selected {len(selected_genes)} genes")
    logger.info(f"  Top 10: {list(selected_genes[:10])}")
    
    # Train final classifier
    X_final = X_hvg[selected_genes].values
    scaler_final = StandardScaler(with_mean=True, with_std=True).fit(X_final)
    X_scaled = scaler_final.transform(X_final)
    
    y_all = metadata['subtype'].astype(str).values
    clf = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=2000,
        class_weight='balanced',
        random_state=RANDOM_STATE
    ).fit(X_scaled, y_all)
    
    train_acc = clf.score(X_scaled, y_all)
    logger.info(f"  Training accuracy: {train_acc:.4f}")
    logger.info(f"  Classes: {clf.classes_}")
    
    model_bundle = {
        'selected_genes': list(selected_genes),
        'hvg_genes': list(hvg_genes),
        'scaler': scaler_final,
        'classifier': clf,
        'classes': list(clf.classes_),
        'topk': topk,
        'n_hvg': n_hvg
    }
    
    return model_bundle


# ============================================================================
# External Validation
# ============================================================================

def validate_external_cohort(model_bundle: Dict,
                            external_counts: pd.DataFrame,
                            external_metadata: pd.DataFrame,
                            cohort_name: str) -> Dict:
    """
    Apply trained model to external validation cohort.
    
    Parameters
    ----------
    model_bundle : Dict
        Trained model bundle.
    external_counts : pd.DataFrame
        External cohort pseudobulk counts.
    external_metadata : pd.DataFrame
        External cohort metadata.
    cohort_name : str
        Name of external cohort.
        
    Returns
    -------
    Dict
        Validation results with predictions and metrics.
    """
    logger.info(f"\nValidating on {cohort_name} cohort")
    
    # Transform to log-CPM
    external_logcpm = log_cpm(external_counts)
    
    # Align to model genes (fill missing with 0)
    selected_genes = model_bundle['selected_genes']
    X_ext = external_logcpm.reindex(columns=selected_genes, fill_value=0.0).values
    
    # Check gene overlap
    overlap = external_logcpm.columns.intersection(selected_genes)
    logger.info(f"  Gene overlap: {len(overlap)}/{len(selected_genes)}")
    
    # Transform and predict
    X_scaled = model_bundle['scaler'].transform(X_ext)
    y_pred = model_bundle['classifier'].predict(X_scaled)
    y_proba = model_bundle['classifier'].predict_proba(X_scaled)
    
    results = {
        'y_pred': y_pred,
        'y_proba': y_proba,
        'gene_overlap': len(overlap),
        'gene_overlap_pct': len(overlap) / len(selected_genes) * 100
    }
    
    # Calculate metrics if true labels available
    if 'subtype' in external_metadata.columns:
        y_true = external_metadata['subtype'].astype(str).values
        acc, bal_acc, f1_macro = evaluate_predictions(y_true, y_pred)
        
        results.update({
            'y_true': y_true,
            'accuracy': acc,
            'balanced_accuracy': bal_acc,
            'f1_macro': f1_macro
        })
        
        logger.info(f"  Accuracy: {acc:.4f}")
        logger.info(f"  Balanced Accuracy: {bal_acc:.4f}")
        logger.info(f"  F1 (Macro): {f1_macro:.4f}")
    
    return results


# ============================================================================
# Main Workflow
# ============================================================================

def main():
    """Main analysis workflow."""
    
    parser = argparse.ArgumentParser(
        description='Histological subtype classification from scRNA-seq data'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing 10X format single-cell data'
    )
    parser.add_argument(
        '--metadata_path',
        type=str,
        required=True,
        help='Path to metadata CSV file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Output directory for results and models'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['multiclass', 'binary', 'both'],
        default='both',
        help='Classification mode'
    )
    parser.add_argument(
        '--n_hvg',
        type=int,
        default=2000,
        help='Number of highly variable genes'
    )
    parser.add_argument(
        '--topk_multiclass',
        type=int,
        nargs='+',
        default=[100],
        help='Number of PC1 genes for multi-class classifier (can specify multiple)'
    )
    parser.add_argument(
        '--topk_binary',
        type=int,
        nargs='+',
        default=[100, 200, 500, 1000],
        help='Number of PC1 genes for binary classifier (can specify multiple)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    metadata_path = Path(args.metadata_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define sample mappings (UPDATE THIS FOR YOUR DATA)
    sample_mapping = {
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
    logger.info("="*70)
    logger.info("Loading Data")
    logger.info("="*70)
    adata = load_single_cell_data(data_dir, metadata_path, sample_mapping)
    
    # ========================================================================
    # Multi-class Classification
    # ========================================================================
    
    if args.mode in ['multiclass', 'both']:
        logger.info("\n" + "="*70)
        logger.info("Multi-Class Classification")
        logger.info("="*70)
        
        # Filter to multi-class samples (e.g., exclude certain groups)
        adata_mc = adata.copy()
        # Add custom filtering logic here if needed
        
        # Create pseudobulk
        pb_counts_mc = pseudobulk_sum(adata_mc, 'sample_id', 'counts')
        meta_mc = (
            pd.Series(pb_counts_mc.index, name='sample_id')
            .to_frame()
            .assign(subtype=lambda df: df['sample_id'].map(sample_mapping))
            .set_index('sample_id')
        )
        pb_log_mc = log_cpm(pb_counts_mc)
        
        logger.info(f"Multi-class samples: {len(meta_mc)}")
        logger.info(f"Subtype distribution:\n{meta_mc['subtype'].value_counts()}")
        
        # PCA visualization
        colors_mc = {
            'Group_A': '#2ecc71',
            'Group_B': '#e74c3c',
            'Group_C': '#f39c12',
            'Group_D': '#3498db'
        }
        visualize_pca(
            adata_mc, pb_log_mc, meta_mc,
            args.n_hvg, 'Multi-Class PCA',
            colors_mc,
            save_path=output_dir / 'pca_multiclass.png'
        )
        
        # Test multiple topk values
        results_mc_all = {}
        models_mc_all = {}
        
        for topk in args.topk_multiclass:
            logger.info(f"\n{'-'*70}")
            logger.info(f"Testing topk={topk}")
            logger.info(f"{'-'*70}")
            
            # LOOCV
            results = run_loocv_classifier(
                adata_mc, pb_counts_mc, meta_mc,
                topk, args.n_hvg,
                f"Multi-Class (top-{topk})"
            )
            results_mc_all[topk] = results
            
            # Train final model
            model = train_final_model(
                adata_mc, pb_log_mc, meta_mc,
                topk, args.n_hvg
            )
            models_mc_all[topk] = model
            
            # Save model
            with open(output_dir / f'model_multiclass_top{topk}.pkl', 'wb') as f:
                pickle.dump(model, f)
            
            # Save genes
            with open(output_dir / f'genes_multiclass_top{topk}.txt', 'w') as f:
                for gene in model['selected_genes']:
                    f.write(f'{gene}\n')
            
            # Save LOOCV results
            loocv_df = pd.DataFrame({
                'sample_id': meta_mc.index,
                'true_label': results['y_true'],
                'predicted_label': results['y_pred']
            })
            loocv_df.to_csv(output_dir / f'loocv_results_multiclass_top{topk}.csv', index=False)
        
        # Summary table
        summary_mc = pd.DataFrame([
            {
                'topk': k,
                'accuracy': v['accuracy'],
                'balanced_accuracy': v['balanced_accuracy'],
                'f1_macro': v['f1_macro']
            }
            for k, v in results_mc_all.items()
        ])
        
        logger.info("\nMulti-Class Classifier Performance Summary:")
        print(summary_mc.to_string(index=False))
        summary_mc.to_csv(output_dir / 'summary_multiclass.csv', index=False)
        
        # Plot confusion matrices
        n_models_mc = len(args.topk_multiclass)
        fig, axes = plt.subplots(1, n_models_mc, figsize=(6*n_models_mc, 5))
        if n_models_mc == 1:
            axes = [axes]
        
        for ax, (topk, results) in zip(axes, results_mc_all.items()):
            labels = sorted(np.unique(results['y_true']))
            cm = confusion_matrix(results['y_true'], results['y_pred'], labels=labels)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                        xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'top-{topk}\nF1: {results["f1_macro"]:.3f}')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_multiclass_all.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # Binary Classification
    # ========================================================================
    
    if args.mode in ['binary', 'both']:
        logger.info("\n" + "="*70)
        logger.info("Binary Classification")
        logger.info("="*70)
        
        # Filter to binary classes (e.g., Group_A vs Group_B)
        adata_bin = adata[adata.obs['subtype'].isin(['Group_A', 'Group_B'])].copy()
        
        # Create pseudobulk
        pb_counts_bin = pseudobulk_sum(adata_bin, 'sample_id', 'counts')
        meta_bin = (
            pd.Series(pb_counts_bin.index, name='sample_id')
            .to_frame()
            .assign(subtype=lambda df: df['sample_id'].map(sample_mapping))
            .set_index('sample_id')
        )
        meta_bin = meta_bin[meta_bin['subtype'].isin(['Group_A', 'Group_B'])]
        pb_counts_bin = pb_counts_bin.loc[meta_bin.index]
        pb_log_bin = log_cpm(pb_counts_bin)
        
        logger.info(f"Binary samples: {len(meta_bin)}")
        logger.info(f"Class distribution:\n{meta_bin['subtype'].value_counts()}")
        
        # PCA visualization
        colors_bin = {'Group_A': '#3498db', 'Group_B': '#e74c3c'}
        visualize_pca(
            adata_bin, pb_log_bin, meta_bin,
            args.n_hvg, 'Binary PCA: Group_A vs Group_B',
            colors_bin,
            save_path=output_dir / 'pca_binary.png'
        )
        
        # Test multiple topk values
        results_bin_all = {}
        models_bin_all = {}
        
        for topk in args.topk_binary:
            logger.info(f"\n{'-'*70}")
            logger.info(f"Testing topk={topk}")
            logger.info(f"{'-'*70}")
            
            # LOOCV
            results = run_loocv_classifier(
                adata_bin, pb_counts_bin, meta_bin,
                topk, args.n_hvg,
                f"Binary (top-{topk})"
            )
            results_bin_all[topk] = results
            
            # Train final model
            model = train_final_model(
                adata_bin, pb_log_bin, meta_bin,
                topk, args.n_hvg
            )
            models_bin_all[topk] = model
            
            # Save model
            with open(output_dir / f'model_binary_top{topk}.pkl', 'wb') as f:
                pickle.dump(model, f)
            
            # Save genes
            with open(output_dir / f'genes_binary_top{topk}.txt', 'w') as f:
                for gene in model['selected_genes']:
                    f.write(f'{gene}\n')
            
            # Save LOOCV results
            loocv_df = pd.DataFrame({
                'sample_id': meta_bin.index,
                'true_label': results['y_true'],
                'predicted_label': results['y_pred']
            })
            loocv_df.to_csv(output_dir / f'loocv_results_binary_top{topk}.csv', index=False)
        
        # Summary table
        summary_bin = pd.DataFrame([
            {
                'topk': k,
                'accuracy': v['accuracy'],
                'balanced_accuracy': v['balanced_accuracy'],
                'f1_macro': v['f1_macro'],
                'auc': v.get('auc', np.nan)
            }
            for k, v in results_bin_all.items()
        ])
        
        logger.info("\nBinary Classifier Performance Summary:")
        print(summary_bin.to_string(index=False))
        summary_bin.to_csv(output_dir / 'summary_binary.csv', index=False)
        
        # Plot confusion matrices
        n_models = len(args.topk_binary)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        for ax, (topk, results) in zip(axes, results_bin_all.items()):
            labels = sorted(np.unique(results['y_true']))
            cm = confusion_matrix(results['y_true'], results['y_pred'], labels=labels)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'top-{topk}\nF1: {results["f1_macro"]:.3f}')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_binary_all.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # Save metrics summary
    # ========================================================================
    
    metrics_summary = {}
    
    if args.mode in ['multiclass', 'both']:
        metrics_summary['multiclass'] = {
            str(k): {
                'accuracy': float(v['accuracy']),
                'balanced_accuracy': float(v['balanced_accuracy']),
                'f1_macro': float(v['f1_macro'])
            }
            for k, v in results_mc_all.items()
        }
    
    if args.mode in ['binary', 'both']:
        metrics_summary['binary'] = {
            str(k): {
                'accuracy': float(v['accuracy']),
                'balanced_accuracy': float(v['balanced_accuracy']),
                'f1_macro': float(v['f1_macro']),
                'auc': float(v.get('auc', np.nan))
            }
            for k, v in results_bin_all.items()
        }
    
    with open(output_dir / 'metrics_summary.json', 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    logger.info("\n" + "="*70)
    logger.info(f"Analysis complete! Results saved to {output_dir}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
