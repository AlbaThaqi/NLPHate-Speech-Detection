# Processing Scripts (organization)

This folder collects all preprocessing and data-quality scripts used to prepare datasets.

Subfolders:
- `analysis_scripts/` - exploratory analysis scripts (visualizations, EDA helpers)
- `data-quality/` - data quality checks, validation and reports
- `data_analysis/` - analytic scripts used in preprocessing and feature inspection
- `outliers/` - scripts to detect and handle outliers
- `scripts/` - general purpose processing scripts (merging, cleaning)
- `skewness/` - scripts to analyze class imbalance and skewness

Guidelines:
- Do not keep large datasets here; keep scripts only. If you have original scripts elsewhere, move them here and update `data/README.md` to record which script produced which processed file.
- Name scripts clearly: `merge_<source>_to_<target>.py`, `clean_<dataset>.py`, `validate_<dataset>.py`.
