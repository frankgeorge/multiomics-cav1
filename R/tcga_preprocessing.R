# TCGA preprocessing: DESeq2 normalization and label extraction
# Requires: TCGAbiolinks, DESeq2, SummarizedExperiment, dplyr, tidyr
library(TCGAbiolinks)
library(DESeq2)
library(SummarizedExperiment)
library(dplyr)
library(tidyr)

project <- "TCGA-BRCA"

message("Querying TCGA RNA-seq (HTSeq counts) for ", project)
query <- GDCquery(
  project = project,
  data.category = "Transcriptome Profiling",
  data.type = "Gene Expression Quantification",
  workflow.type = "HTSeq - Counts"
)
GDCdownload(query)
rna_data <- GDCprepare(query)

count_data <- assay(rna_data)
col_data <- colData(rna_data)
colnames(count_data) <- col_data$barcode

message("Running DESeq2 size factor estimation and normalization...")
dds <- DESeqDataSetFromMatrix(countData = count_data,
                              colData = col_data,
                              design = ~ 1)
dds <- estimateSizeFactors(dds)
normalized_counts <- counts(dds, normalized = TRUE)

write.csv(as.data.frame(normalized_counts), "normalized_counts.csv", row.names = TRUE)
message("Saved normalized_counts.csv")

message("Querying clinical data...")
clinical_query <- GDCquery(
  project = project,
  data.category = "Clinical",
  data.type = "Clinical Supplement",
  data.format = "BCR Biotab"
)
GDCdownload(clinical_query)
clinical_data <- GDCprepare(clinical_query)

# clinical_patient_brca contains patient-level info
if (!is.null(clinical_data$clinical_patient_brca)) {
  clinical_labels <- clinical_data$clinical_patient_brca %>%
    select(bcr_patient_barcode,
           er_status_by_ihc,
           pr_status_by_ihc,
           her2_status_by_ihc,
           tumor_stage) %>%
    dplyr::rename(sample_id = bcr_patient_barcode,
                  ER_status = er_status_by_ihc,
                  PR_status = pr_status_by_ihc,
                  HER2_status = her2_status_by_ihc)
  write.csv(clinical_labels, "clinical_labels.csv", row.names = FALSE)
  message("Saved clinical_labels.csv")
} else {
  message("Warning: clinical_patient_brca not found in prepared clinical data.")
}

# Extract CAV1 expression (if present) and merge
if ("CAV1" %in% rownames(normalized_counts)) {
  cav1_expr <- as.numeric(normalized_counts["CAV1", ])
} else {
  cav1_expr <- rep(NA, ncol(normalized_counts))
}
merged <- data.frame(sample_id = colnames(normalized_counts),
                     Cav1_expression = cav1_expr,
                     stringsAsFactors = FALSE)
if (exists("clinical_labels")) {
  merged_full <- merge(merged, clinical_labels, by = "sample_id", all.x = TRUE)
} else {
  merged_full <- merged
}
write.csv(merged_full, "merged_tcga_data.csv", row.names = FALSE)
message("Saved merged_tcga_data.csv")
