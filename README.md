# multiomics-cav1

Ready-to-deploy repository for a Cav-1 context prediction demo.
Includes:
- R/tcga_preprocessing.R : DESeq2 normalization & TCGA clinical extraction
- app/ : Flask dashboard + model training example
- vercel.json : Vercel deployment config

## Quickstart (local)

### 1) R preprocessing (requires R and Bioconductor packages)
Run in R:
```
Rscript R/tcga_preprocessing.R
```
This will produce:
- normalized_counts.csv
- clinical_labels.csv
- merged_tcga_data.csv

### 2) Train demo model (Python)
```
python3 app/model/multiomics_cav1_prediction.py
```
This will save a demo PyTorch model to `app/model/models/cav1_model.pth`

### 3) Run the Flask dashboard (local)
```
cd app
pip install -r requirements.txt
python app.py
```
Open http://localhost:5000 and use the dashboard.

## Deploy to Vercel
1. Create a GitHub repo and push this project.
2. Connect the repo to Vercel.
3. Vercel will detect `app/app.py` per vercel.json and deploy the Flask app.

## Notes
- The repo includes demo placeholder model code. For production use, replace the demo proxy labels with curated functional labels for Cav-1 and retrain on GPU/Colab.
- The dashboard uses a demo SHAP-like bar chart. After adding a real SHAP server endpoint, update the `/predict` response to include feature importances.

## License
MIT
