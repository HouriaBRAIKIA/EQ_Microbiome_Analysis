# Explainable AI for Marine Ecological Quality Prediction  
**Integrating Microbiome Data, Metadata, and Diversity**

## 🧠 Description

This project accompanies the forthcoming publication:

**"Explainable AI for Marine Ecological Quality Prediction: Integrating Microbiome Data, Metadata, and Diversity"**

Assessing ecological quality (EQ) is crucial for marine biodiversity monitoring. With the advent of High-Throughput Sequencing technologies, metabarcoding has enabled large-scale microbial community analysis through Operational Taxonomic Unit (OTU) tables.

This study integrates microbiome data, environmental metadata (e.g., pH, salinity, temperature), and diversity indices (alpha and beta) into an explainable machine learning framework for EQ prediction. Using SHapley Additive Explanations (SHAP), we assess feature contributions to model predictions across five genetic markers (V1V2, V3V4, V4, 37F, and V9).

Our results demonstrate that while OTU-based models are highly predictive, incorporating metadata and diversity metrics improves accuracy and interpretability for certain markers. This work aims to enhance trust and transparency in AI-driven biomonitoring.

---

## 📊 Dataset

The data used in this project comes from the publication:

> Cordier, T., Forster, D., Dufresne, Y., Martins, C. I. M., Stoeck, T., & Pawlowski, J. (2018).  
> *Supervised machine learning outperforms taxonomy-based environmental DNA metabarcoding applied to biomonitoring*.  
> Molecular Ecology Resources  
> DOI: [10.1111/1755-0998.12926](https://doi.org/10.1111/1755-0998.12926)

📥 Available at: [https://zenodo.org/record/1286477](https://zenodo.org/record/1286477)

---

## 📁 Project Structure

project/
├── src/ # Python source code
│   ├── normalization.py
│   ├── modeling.py
│   ├── diversity.py
│   ├── multivariate_analysis.py
|   ├── data_preprocessing.py
│   └── SVD.py
├── notebook/ # Example usage notebooks
│   └── OTU_meta_div.ipynb
├── Data/ # Datasets from Cordier et al. (2018)
├── results/ # Output plots and figures
└── README.md


---

## ⚙️ Installation Instructions

### 📦 Python

```bash
pip install -Iv rpy2==3.4.2
pip install pandas==1.5.3
pip install numpy matplotlib seaborn scikit-learn shap scikit-bio

---

⚠️ After installing rpy2, restart your Python kernel or runtime (important for Jupyter users).

📦 R packages (via rpy2)
In your Python code or notebook, execute:

import rpy2.robjects as ro
ro.r('''
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("edgeR")
BiocManager::install("metagenomeSeq")
''')


These are required for normalization:

edgeR: for TMM normalization

---

🧪 How to Use
- Place your OTU tables and metadata in the Data/ folder.

- Explore and test the full analysis pipeline using:

    notebook/OTU_meta.ipynb

- It includes:
    * OTU normalization (TMM and CSS via R)
    * Calculation of alpha diversity indices
    * Random Forest model training
    * SHAP-based feature explanation
    * CCA multivariate analysis

---

🧠 Technologies Used
    * Python
        * pandas, numpy, scikit-learn, matplotlib, seaborn
        * shap, scikit-bio, rpy2
    * R (via rpy2)
        * edgeR, metagenomeSeq for normalization techniques

---

🤝 Contributions
Contributions are welcome!

📧 Contact
For more information, please contact: houria.braikia@dauphine.psl.eu 






               
