# Predicting 30-Day Readmission and Length of Stay in Diabetic Patients

This repository contains a machine-learning project for predicting hospital readmission risk and length of stay in diabetic patients.

The project was designed as an applied clinical ML workflow: data preprocessing, feature engineering, model training, evaluation, and lightweight deployment through API and web interfaces.

## Project summary

- **Problem:** Predict 30-day hospital readmission risk and length of stay for diabetic patients.
- **Domain:** Healthcare AI, clinical prediction modelling, risk stratification.
- **Data:** Structured clinical dataset of diabetic hospital encounters.
- **Methods:** Classical machine learning and deep-learning models, including MLP and transformer-based approaches.
- **Deployment:** FastAPI endpoint, Streamlit application, and simple HTML frontend.

## Why this project matters

Hospital readmission and prolonged length of stay are important healthcare-quality and resource-planning problems. Predictive modelling can help identify high-risk patients earlier, support triage, and guide follow-up planning.

This project is not intended as a clinical product. It is a technical demonstration of a reproducible healthcare-ML pipeline from model development to simple deployment.

## Repository structure

```text
├── DDLS_Enhanced_MemoryOptimized.ipynb      # Main analysis, preprocessing, model training, evaluation
├── app_fastapi.py                           # FastAPI application for model serving
├── app_streamlit.py                         # Streamlit app for interactive use
├── simple_frontend.html                     # Minimal frontend for API interaction
├── requirements.txt                         # Python dependencies
├── best_mlp.pth                             # Saved MLP model weights
├── best_multitask_model.pth                 # Saved multitask model weights
├── best_transformer.pth                     # Saved transformer model weights
└── models/                                  # Preprocessors, model configs, trained model files
```

## Technical focus

The project demonstrates:

- clinical-data preprocessing,
- tabular machine learning,
- deep-learning model development,
- multitask prediction,
- model persistence,
- API deployment with FastAPI,
- interactive prototyping with Streamlit.

## Reproducibility

Clone the repository:

```bash
git clone https://github.com/argyrisker/Predicting-30-Day-Readmission-and-Length-of-Stay-in-Diabetic-Patients.git
cd Predicting-30-Day-Readmission-and-Length-of-Stay-in-Diabetic-Patients
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook DDLS_Enhanced_MemoryOptimized.ipynb
```

Run the Streamlit app:

```bash
streamlit run app_streamlit.py
```

Run the FastAPI app:

```bash
uvicorn app_fastapi:app --reload
```

## Limitations

- The project is intended for educational and research-portfolio purposes.
- External validation would be required before any clinical use.
- Model performance depends on dataset quality, preprocessing choices, and feature availability.
- Clinical deployment would require calibration, fairness analysis, interpretability, monitoring, and medical review.

## Technical keywords

`healthcare-ai` · `clinical-machine-learning` · `readmission-prediction` · `length-of-stay` · `tabular-deep-learning` · `transformer` · `fastapi` · `streamlit` · `python` · `pytorch`

## Author

**Argyrios Kerezis**  
Biomedical AI / healthcare machine learning  
GitHub: [@argyrisker](https://github.com/argyrisker)
