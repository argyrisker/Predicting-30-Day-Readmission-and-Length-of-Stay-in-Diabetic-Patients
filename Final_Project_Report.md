# Final Project Report: Predicting Hospital Readmission for Diabetic Patients

**Author:** Argyrios Kerezis
**Date:** October 22, 2025

## Abstract (≤100 words)

This project addresses the critical challenge of predicting 30-day hospital readmissions for patients with diabetes. Using a large clinical dataset of over 70,000 patient records, we developed and evaluated several machine learning models, including Logistic Regression, XGBoost, and deep learning models like Multi-Task MLP and Transformer. Our best performing model achieved an AUC of [insert best AUC from your notebook] on the test set. The project demonstrates the potential of machine learning to identify high-risk patients, enabling targeted interventions to reduce readmission rates and improve patient outcomes.

## 1. Background and Motivation

Hospital readmissions, particularly for chronic conditions like diabetes, are a major concern for healthcare systems worldwide. They are associated with increased healthcare costs, reduced quality of life for patients, and are often considered an indicator of poor quality of care. The U.S. Center for Medicare and Medicaid Services (CMS) has implemented the Hospital Readmissions Reduction (HRR) Program to penalize hospitals with excess readmission rates for certain conditions, including diabetes.

The motivation for this project was to leverage machine learning to build a predictive model that can identify diabetic patients at high risk of 30-day readmission. By accurately predicting readmission risk, healthcare providers can implement targeted interventions, such as enhanced discharge planning, post-discharge follow-up, and patient education, to reduce the likelihood of readmission. This can lead to improved patient outcomes, reduced healthcare costs, and better alignment with value-based care initiatives like the HRR Program.

## 2. Dataset Summary

**Data Source:**
The dataset used in this project is the "Diabetes 130-US hospitals for years 1999-2008" dataset from the UCI Machine Learning Repository. It was collected by Beata Strack et al. and represents ten years of clinical care at 130 US hospitals and integrated delivery networks. The dataset contains 101,766 encounters for patients with diabetes.

**Preprocessing:**
The raw dataset underwent several preprocessing steps to prepare it for modeling:
*   **Data Cleaning:** Missing values, represented as "?", were replaced with NaN. Rows with missing values in key diagnostic columns and invalid data (e.g., expired patients) were removed.
*   **Feature Engineering:**
    *   Categorical features like `admission_type_id`, `discharge_disposition_id`, and `admission_source_id` were re-encoded into fewer, more meaningful categories.
    *   The `age` feature was encoded as a numeric feature.
    *   To prevent data leakage, only the first encounter for each patient was kept.
    *   Clinical risk scores like the Charlson Index, LACE Index, and a custom hospital score were engineered from existing features.
*   **Outlier Removal:** Outliers in numeric features were removed using the z-score method.
*   **Encoding:** Categorical features were one-hot encoded.
*   **Scaling:** Numerical features were standardized using `StandardScaler`.

**Data Splits:**
The preprocessed dataset was split into training (70%), validation (15%), and test (15%) sets. The splits were stratified based on the target variable (`readmitted`) to maintain the same class distribution in each set.

**Distributions:**
The dataset is imbalanced, with a readmission rate of approximately 11%. To address this, the training data was balanced using the Synthetic Minority Over-sampling TEchnique (SMOTE).

## 3. Method Description

**Workflow:**
The project followed a standard machine learning workflow:
1.  Data loading and exploration.
2.  Data cleaning and preprocessing.
3.  Feature engineering.
4.  Model training and hyperparameter tuning.
5.  Model evaluation and comparison.
6.  Interpretation of the best performing model using SHAP.

**Models:**
Several models were trained and evaluated:
*   **Logistic Regression:** A baseline linear model.
*   **XGBoost:** A gradient boosting model known for its performance.
*   **Multi-Task MLP:** A Multi-Layer Perceptron designed to simultaneously predict readmission and length of stay.
*   **Transformer:** A Transformer-based model, also for the multi-task prediction.

**Evaluation Metrics:**
The models were evaluated using the following metrics:
*   **Area Under the Receiver Operating Characteristic Curve (AUC):** To assess the model's ability to distinguish between classes.
*   **F1-Score, Precision, and Recall:** To evaluate the model's performance on the imbalanced dataset.
*   **Mean Absolute Error (MAE) and R-squared (R2):** For the length of stay prediction task.

## 4. Results

The performance of the models on the test set is summarized below.

**Readmission Prediction Results:**
| Model                | AUC       | F1    |
| -------------------- | ------    | ----- |
                   
| LogisticRegression    | 0.710593 |     0.693112 |
| XGBoost               | 0.955466 |     0.9270   |

**Length of Stay Prediction Results:**
| Model          | MAE    | R2     |
| -------------- | ------ | ------ |
| MultiTaskMLP   | [From your notebook] |        |
| TransformerMTL |        |        |


## 5. Conclusion & Discussion

**Findings:**
In this project, we successfully developed and evaluated several machine learning models for predicting 30-day hospital readmission for diabetic patients. The [mention your best model] achieved the best performance with an AUC of [best AUC]. The most important predictors for readmission were [mention top features].

**Limitations:**
This study has some limitations. The dataset is from 1999-2008, and clinical practices may have changed since then. The performance of the models, while promising, is not perfect, and there is still room for improvement. The dataset also has a significant number of missing values in some columns, which required imputation or removal.

**Future Directions:**
Future work could focus on:
*   Using more recent data to build more up-to-date models.
*   Exploring more advanced modeling techniques, such as deep learning architectures specifically designed for healthcare data.
*   Incorporating additional data sources, such as unstructured clinical notes, to improve prediction accuracy.
*   Developing a real-time prediction system that can be integrated into a clinical workflow to support decision-making.

## 6. Data and Code Availability

*   **Data:** The dataset is publicly available from the UCI Machine Learning Repository: [https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)
*   **Code:** The code for this project is available at [Link to your GitHub repository].

## 7. Acknowledgments

I would like to thank [mention any contributions or support you received].

*Note on GenAI tools:* Generative AI tools were used to assist with code generation, debugging, and report writing in this project.

## 8. References

*   Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records. *BioMed research international*, *2014*.
*   The Hospital Readmissions Reduction (HRR) Program, Center for Medicare and Medicaid Services. Website: [https://www.cms.gov/Medicare/Quality-Initiatives-Patient-Assessment-Instruments/Value-Based-Programs/HRRP/Hospital-Readmission-Reduction-Program](https://www.cms.gov/Medicare/Quality-Initiatives-Patient-Assessment-Instruments/Value-Based-Programs/HRRP/Hospital-Readmission-Reduction-Program)
*   Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2022). PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3). *PhysioNet*. RRID:SCR_007345. [https://doi.org/10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45)
