#!/usr/bin/env Rscript
# ==============================================================================
# Macrophage - reference Genesets/Pathways Activity Analysis (Signature score)
# ==============================================================================

# Load required libraries
suppressPackageStartupMessages({
  library(Seurat)
  library(msigdbr)
  library(ggplot2)
  library(dplyr)
})

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Define output directory
OUTPUT_DIR <- "/path/to/output"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# Define analysis parameters (UPDATE THESE FOR YOUR DATA)
MACROPHAGE_SUBTYPES <- c("Macrophage_Type1", "Macrophage_Type2")  # Cell types to analyze
PATHWAY_NAME <- "GOBP_POSITIVE_REGULATION_OF_CHOLESTEROL_EFFLUX"  # Example pathway from MSigDB
CELL_TYPE_COLUMN <- "cell_type.lv0"      # Metadata column for main cell type
SUBTYPE_COLUMN <- "cell_type.lv1"     # Metadata column for cell subtype
GROUP_COLUMN <- "group"              # Metadata column for experimental groups
BATCH_COLUMN <- "sample_id"          # Metadata column for batch/sample ID

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

#' Load MSigDB gene sets
#' 
#' Downloads and organizes MSigDB gene sets for a specified species.
#' 
#' @param species Species name (default: "Homo sapiens")
#' @return Named list of gene sets
#' @examples
#' gene_sets <- load_msigdb_genesets()
load_msigdb_genesets <- function(species = "Homo sapiens") {
  
  cat("Loading MSigDB gene sets for", species, "...\n")
  
  # Download MSigDB data
  msigdb_data <- msigdbr(species = species)
  
  # Convert to named list format (gene set name -> gene symbols)
  geneset_list <- split(
    x = msigdb_data$gene_symbol,
    f = msigdb_data$gs_name
  )
  
  cat("  Loaded", length(geneset_list), "gene sets\n\n")
  
  return(geneset_list)
}

#' Calculate pathway enrichment score
#' 
#' Computes module score for a gene set and returns z-score normalized values.
#' 
#' @param so Seurat object
#' @param geneset_list Named list of gene sets
#' @param pathway_name Name of pathway to score
#' @param score_name Output column name for score
#' @return Seurat object with added pathway score
#' @examples
#' so <- calculate_pathway_score(so, gene_sets, "GOBP_APOPTOSIS", "apoptosis_score")
calculate_pathway_score <- function(so, 
                                   geneset_list, 
                                   pathway_name,
                                   score_name = "pathway_score") {
  
  if (!pathway_name %in% names(geneset_list)) {
    stop(sprintf("Pathway '%s' not found in gene set list", pathway_name))
  }
  
  cat("Calculating pathway score for:", pathway_name, "\n")
  
  # Normalize data if needed
  if (!"RNA" %in% names(so@assays)) {
    so <- NormalizeData(so)
  }
  
  # Add module score
  so <- AddModuleScore(
    so,
    features = geneset_list[pathway_name],
    name = paste0(pathway_name, "_")
  )
  
  # Get the automatically generated column name
  auto_name <- paste0(pathway_name, "_1")
  
  # Z-score 
  so@meta.data[[score_name]] <- scale(
    so@meta.data[[auto_name]]
  )[, 1]
  
  # Remove temporary column
  so@meta.data[[auto_name]] <- NULL
  
  cat("  Score added as column:", score_name, "\n\n")
  
  return(so)
}

#' Test pathway score differences across groups
#' 
#' Performs ANOVA followed by Tukey HSD
#' post-hoc test to compare pathway scores across experimental groups.
#' 
#' @param so Seurat object with pathway scores
#' @param score_column Name of pathway score column
#' @param group_column Name of group/condition column
#' @param batch_column Name of sample column for correction
#' @return List containing ANOVA model and Tukey HSD results
#' @examples
#' results <- test_pathway_differences(so, "pathway_score", "condition", "sample")
test_pathway_differences <- function(so,
                                    score_column,
                                    group_column,
                                    sample_column) {
  
  cat("Testing pathway score differences across groups...\n")
  cat("  Score:", score_column, "\n")
  cat("  Groups:", group_column, "\n")
  cat("  Sample correction:", sample_column, "\n\n")
  
  # Extract required columns
  test_data <- so@meta.data[, c(group_column, sample_column, score_column)]
  colnames(test_data) <- c("group", "sample", "score")
  
  # Build ANOVA model 
  formula_str <- "score ~ group + sample"
  aov_model <- aov(as.formula(formula_str), data = test_data)
  
  # Post-hoc Tukey HSD test
  tukey_results <- TukeyHSD(aov_model)
  
  # Print results
  cat("ANOVA Results:\n")
  print(summary(aov_model))
  cat("\n")
  
  cat("Tukey HSD Post-hoc Test (Group Comparisons):\n")
  print(tukey_results$group)
  cat("\n")
  
  return(list(
    model = aov_model,
    tukey = tukey_results,
    group_comparisons = tukey_results$group
  ))
}


# ==============================================================================
# EXAMPLE WORKFLOW
# ==============================================================================

# Step 1: Load MSigDB gene sets
geneset_list <- load_msigdb_genesets()

# Step 2: Subset to macrophage cells
Idents(so) <- CELL_TYPE_COLUMN
macrophage <- subset(so, idents = MACROPHAGE_SUBTYPES)

# Step 3: Calculate pathway enrichment score
macrophage <- calculate_pathway_score(
  macrophage,
  geneset_list,
  pathway_name = PATHWAY_NAME,
  score_name = "pathway_score"
)

# Step 4: Statistical testing (ANOVA with batch correction + Tukey HSD)
stat_results <- test_pathway_differences(
  macrophage,
  score_column = "pathway_score",
  group_column = GROUP_COLUMN,
  batch_column = BATCH_COLUMN
)

