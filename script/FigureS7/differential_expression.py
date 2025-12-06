#!/usr/bin/env python3
"""
Differential expression (DE) analysis script.

- A 10X directory containing matrix.mtx, barcodes.tsv(.gz), features/genes.tsv(.gz)
- A metadata CSV with at least columns: sample_id, subtype 

Key features:
- Counts layer enforcement with integer checks
- Pseudobulk generation by summation per sample (min cells configurable)
- Transfer counts + sample metadata to R with rpy2
- Four DESeq2 pairwise comparisons
- Export results to output directory

Usage example:
    python DE_script.py \
        --data-dir /path/to/10x_dir \
        --metadata /path/to/metadata.csv \
        --sample-col sample_id \
        --subtype-col subtype \
        --min-cells 10 \
        --output-dir ./results/deg_pairwise \
        [--subtype-map /path/to/subtype_map.csv]

"""
from __future__ import annotations
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

import anndata as ad
import scanpy as sc

# rpy2 imports 
import rpy2.robjects as ro
import rpy2.robjects.conversion as conversion
import rpy2.robjects.pandas2ri as pandas2ri
from rpy2.robjects.vectors import StrVector

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# -------------------------
# Utility functions
# -------------------------

def _to_csr(matrix) -> sp.csr_matrix:
    return matrix if sp.isspmatrix_csr(matrix) else sp.csr_matrix(matrix)


def ensure_counts_layer(adata: ad.AnnData):
    if "counts" in adata.layers:
        adata.layers["counts"] = _to_csr(adata.layers["counts"])
        logging.info("Using existing adata.layers['counts']")
    else:
        X = _to_csr(adata.X)
        adata.layers["counts"] = X.copy()
        logging.info("Created adata.layers['counts'] from adata.X")


def check_counts_integer(adata: ad.AnnData) -> bool:
    counts = adata.layers["counts"] if "counts" in adata.layers else adata.X
    if sp.issparse(counts):
        is_integer = counts.data.size == 0 or np.all(np.equal(counts.data, np.floor(counts.data)))
    else:
        is_integer = np.all(np.equal(counts, np.floor(counts)))
    logging.info(f"Counts are integer: {is_integer}")
    return is_integer


# -------------------------
# Pseudobulk
# -------------------------

def make_pseudobulk(
    adata: ad.AnnData,
    sample_col: str,
    subtype_col: str,
    min_cells: int = 10,
) -> ad.AnnData:
    counts = adata.layers["counts"]
    if not sp.issparse(counts):
        counts = sp.csr_matrix(counts)

    samples = adata.obs[sample_col].values
    sample_ids = np.unique(samples)

    pb_data = []
    pb_obs = []

    for sid in sample_ids:
        mask = samples == sid
        n_cells = mask.sum()
        if n_cells < min_cells:
            logging.info(f"Skipping sample {sid}: only {n_cells} cells (< {min_cells})")
            continue

        sample_counts = counts[mask, :].sum(axis=0)
        pb_data.append(np.asarray(sample_counts).flatten())

        sample_meta = adata.obs[mask].iloc[0][[subtype_col]].to_dict()
        sample_meta["sample_id"] = sid
        sample_meta["n_cells"] = n_cells
        pb_obs.append(sample_meta)

    if len(pb_data) == 0:
        raise ValueError("No samples passed min_cells threshold; cannot build pseudobulk.")

    pb_counts = np.vstack(pb_data)
    pb_obs_df = pd.DataFrame(pb_obs)
    pb_obs_df.index = pb_obs_df["sample_id"]

    adata_pb = ad.AnnData(X=sp.csr_matrix(pb_counts), obs=pb_obs_df, var=adata.var.copy())
    adata_pb.layers["counts"] = adata_pb.X.copy()
    logging.info(f"Created pseudobulk: {adata_pb.shape[0]} samples × {adata_pb.shape[1]} genes")
    return adata_pb


# -------------------------
# DESeq2 in R
# -------------------------

def run_deseq2_pairwise(count_mat: pd.DataFrame, coldata: pd.DataFrame, subtype_col: str, output_dir: Path, alpha: float = 0.05):
    converter = conversion.Converter("pandas+default")
    converter += pandas2ri.converter

    with ro.conversion.localconverter(converter):
        ro.globalenv["count_mat"] = count_mat
        ro.globalenv["coldata"] = coldata
    ro.globalenv["subtype_col"] = StrVector([subtype_col])[0]

    # R code: setup
    ro.r('''
        suppressMessages(library(DESeq2))
        suppressMessages(library(apeglm))
        suppressMessages(library(dplyr))
        stopifnot(identical(colnames(count_mat), rownames(coldata)))
        coldata[[subtype_col]] <- factor(coldata[[subtype_col]])
        alpha <- ''' + str(alpha) + '''
        comparisons <- list(
          AP_vs_MP = c("AP", "MP"),
          MP_vs_Solid = c("MP", "Solid"),
          AP_vs_Solid = c("AP", "Solid"),
          AP_vs_APSolid = c("AP", "AP+Solid")
        )

        all_results <- list()
        all_results_shr <- list()
        deg_summary <- data.frame()

        for (comp_name in names(comparisons)) {
          groups <- comparisons[[comp_name]]
          group1 <- groups[1]
          group2 <- groups[2]

          coldata_work <- coldata
          if (comp_name %in% c("AP_vs_Solid", "AP_vs_MP")) {
            coldata_work[[subtype_col]] <- as.character(coldata_work[[subtype_col]])
            coldata_work[[subtype_col]][coldata_work[[subtype_col]] == "AP+Solid"] <- "AP"
            coldata_work[[subtype_col]] <- factor(coldata_work[[subtype_col]])
          }

          samples_subset <- coldata_work[[subtype_col]] %in% groups
          coldata_subset <- droplevels(coldata_work[samples_subset, , drop = FALSE])
          count_mat_subset <- count_mat[, rownames(coldata_subset), drop = FALSE]
          coldata_subset[[subtype_col]] <- relevel(factor(coldata_subset[[subtype_col]]), ref = group2)

          keep <- rowSums(count_mat_subset > 0) >= 1
          count_mat_filt <- count_mat_subset[keep, , drop = FALSE]

          dds <- DESeqDataSetFromMatrix(countData = count_mat_filt, colData = coldata_subset, design = as.formula(paste0("~ ", subtype_col)))
          dds <- DESeq(dds, quiet = TRUE)
          coef_name <- resultsNames(dds)[2]
          res <- results(dds, name = coef_name, alpha = alpha)
          res_shr <- lfcShrink(dds, coef = 2, type = "apeglm", res = res)

          n_up <- sum(res$padj < alpha & res_shr$log2FoldChange > 0, na.rm = TRUE)
          n_down <- sum(res$padj < alpha & res_shr$log2FoldChange < 0, na.rm = TRUE)
          n_total <- n_up + n_down

          all_results[[comp_name]] <- res
          all_results_shr[[comp_name]] <- res_shr
          deg_summary <- rbind(deg_summary, data.frame(
            comparison = comp_name,
            n_genes_tested = nrow(res),
            n_DEG_total = n_total,
            n_DEG_up = n_up,
            n_DEG_down = n_down,
            stringsAsFactors = FALSE
          ))
        }

        output_dir <- ''' + str(output_dir.as_posix()) + '''
        dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
                # Export CSVs
        for (comp_name in names(all_results)) {
          res_df <- as.data.frame(all_results[[comp_name]]); res_df$gene <- rownames(res_df)
          res_shr_df <- as.data.frame(all_results_shr[[comp_name]]); res_shr_df$gene <- rownames(res_shr_df)
          res_final <- merge(
            res_df[, c("gene", "baseMean", "log2FoldChange", "lfcSE", "pvalue", "padj")],
            res_shr_df[, c("gene", "log2FoldChange", "lfcSE")],
            by = "gene",
            suffixes = c("_unshrunken", "_shrunken")
          )
          res_final <- res_final[, c("gene", "baseMean", "log2FoldChange_unshrunken", "lfcSE_unshrunken", "log2FoldChange_shrunken", "lfcSE_shrunken", "pvalue", "padj")]
          res_final <- res_final[order(res_final$padj), ]
          outfile <- file.path(output_dir, paste0(comp_name, "_full_results.csv"))
          write.csv(res_final, outfile, row.names = FALSE, quote = FALSE)
          res_sig <- res_final[!is.na(res_final$padj) & res_final$padj < alpha, ]
          outfile_sig <- file.path(output_dir, paste0(comp_name, "_DEGs_padj", alpha, ".csv"))
          write.csv(res_sig, outfile_sig, row.names = FALSE, quote = FALSE)
        }

        # Save summary
        summary_file <- file.path(output_dir, "DEG_summary.csv")
        write.csv(deg_summary, summary_file, row.names = FALSE, quote = FALSE)
    ''')


# -------------------------
# Main
# -------------------------

def load_subtype_map(csv_path: Optional[Path]) -> Optional[Dict[str, str]]:
    if not csv_path:
        return None
    df = pd.read_csv(csv_path)
    if not {"sample_id", "subtype_label"}.issubset(df.columns):
        raise ValueError("Subtype map must contain columns: sample_id, subtype_label")
    return dict(zip(df["sample_id"].astype(str), df["subtype_label"].astype(str)))


def main():
    parser = argparse.ArgumentParser(description="Anonymized DEG pipeline for histologic subtypes")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to 10X data directory")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to metadata CSV")
    parser.add_argument("--sample-col", type=str, default="sample_id", help="Column name for sample IDs in metadata")
    parser.add_argument("--subtype-col", type=str, default="subtype", help="Column name for subtype in metadata")
    parser.add_argument("--min-cells", type=int, default=10, help="Minimum cells per sample to include")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write results")
    parser.add_argument("--subtype-map", type=Path, default=None, help="Optional CSV: sample_id,subtype_label mapping")
    args = parser.parse_args()

    # Load 10X data
    logging.info(f"Reading 10X data from: {args.data_dir}")
    adata = sc.read_10x_mtx(args.data_dir, var_names="gene_symbols", cache=True)

    # Load metadata
    logging.info(f"Reading metadata: {args.metadata}")
    metadata = pd.read_csv(args.metadata)
    if args.sample_col not in metadata.columns:
        raise ValueError(f"Metadata missing sample column '{args.sample_col}'")
    if args.subtype_col not in metadata.columns:
        logging.warning(f"Metadata missing subtype column '{args.subtype_col}'; will rely on subtype map if provided.")
        metadata[args.subtype_col] = pd.NA

    # Align metadata to adata
    if metadata.index.name is None or metadata.index.name == "":
        # Use barcodes as index if present
        if "barcode" in metadata.columns:
            metadata = metadata.set_index("barcode")
        else:
            # fallback: assume rows correspond to adata.obs order
            metadata.index = adata.obs_names
    metadata = metadata.loc[adata.obs_names, :].copy()

    # Attach metadata
    adata.obs = metadata.copy()

    # Optional subtype mapping override
    subtype_map = load_subtype_map(args.subtype_map)
    if subtype_map:
        logging.info("Applying subtype mapping override from CSV")
        adata.obs[args.subtype_col] = adata.obs[args.sample_col].astype(str).map(subtype_map).fillna(adata.obs[args.subtype_col])

    # Enforce counts layer
    ensure_counts_layer(adata)
    check_counts_integer(adata)

    # Work on cancer cells as provided (assumed already filtered)
    adata_cancer = adata.copy()
    logging.info(f"Total cells: {adata_cancer.shape[0]} × {adata_cancer.shape[1]} genes")

    # Pseudobulk
    adata_pb = make_pseudobulk(
        adata_cancer,
        sample_col=args.sample_col,
        subtype_col=args.subtype_col,
        min_cells=args.min_cells,
    )

    # Count matrix: genes x samples
    Xpb = adata_pb.X
    counts_np = Xpb.T.toarray() if sp.issparse(Xpb) else np.asarray(Xpb.T)
    counts_np = counts_np.astype(np.int64, copy=False)
    count_mat = pd.DataFrame(
        counts_np,
        index=adata_pb.var_names.astype(str),
        columns=adata_pb.obs["sample_id"].astype(str)
    )

    # Sample metadata
    coldata = adata_pb.obs.copy()
    coldata = coldata.set_index("sample_id")
    coldata = coldata.loc[count_mat.columns]

    if not np.all(np.equal(coldata.index.astype(str), count_mat.columns.astype(str))):
        raise ValueError("Mismatch between coldata.index and count_mat.columns")

    # Run DESeq2 analyses
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Running DESeq2 pairwise comparisons; output: {output_dir}")
    run_deseq2_pairwise(count_mat, coldata, args.subtype_col, output_dir)

    logging.info("Analysis complete. Outputs written to:")
    logging.info(output_dir)


if __name__ == "__main__":
    main()
