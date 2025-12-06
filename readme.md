# Single-cell and spatial transcriptomics analysis of histologic subtypes in Lung Adenocarcinoma

This repository contains code and analysis pipelines for a study of the cellular and spatial architecture underlying histologic subtypes of lung adenocarcinoma (LUAD), based on single-cell RNA sequencing (scRNA-seq) and spatial transcriptomics (10x Visium).

---

## Summary

### Background
Tumor histology reflects disease aggressiveness and clinical outcomes in cancer patients. Lung adenocarcinomas (LUADs) are classified based on predominant histologic patterns, including high-grade micropapillary and solid subtypes, which are associated with unfavorable clinical features and prognosis. However, the cellular and molecular features underlying these histologic subtypes remain poorly understood.

### Methods
We used scRNA-seq and spatial transcriptomics (10x Visium) to profile **117,266 cells** from **18 treatment-naïve LUADs** with heterogeneous histologic patterns. By integrating single-cell transcriptional states with spatial information, we characterized the cellular identity and spatial organization driving LUAD heterogeneity.

### Results
Our analyses show that histologic subtypes can be distinguished by:

- subtype-specific cancer cell subpopulations, and  
- immunosuppressive phenotypes in the tumor microenvironment (TME).

We dissect intercellular interactions among cancer cells, macrophages, and CD8⁺ T cells in the prognostically unfavorable solid subtype, revealing how these interactions drive cancer cell plasticity and promote an immunosuppressive TME. Furthermore, we identify **HMGA1** as a potential clinically relevant biomarker and therapeutic target for solid-subtype LUAD.


---

## Interactive Data Exploration (cellxgene)

Interactive visualization of selected processed datasets is available via **cellxgene**:

👉 [Open in cellxgene] TBA


---

## Requirements

The analysis uses both **R** and **Python**.

### R

- R ≥ 4.3  
- Core packages
  - `Seurat`, `SeuratObject`
  - `Harmony`
  - `dplyr`, `data.table`
  - `ggplot2`, `patchwork`

### Python

- Python ≥ 3.9  
- Core packages
  - `numpy`, `pandas`, `scikit-learn`
  - `anndata`, `scanpy`, `celltypist`
  - `matplotlib`, `seaborn`
  

---
