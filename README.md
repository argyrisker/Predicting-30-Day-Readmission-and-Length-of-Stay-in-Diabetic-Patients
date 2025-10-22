# Predicting Hospital Readmission for Diabetic Patients

This project aims to predict 30-day hospital readmissions for patients with diabetes using machine learning. The project is based on a large clinical dataset and explores various models to identify high-risk patients.

## Folder Structure

```
├── app_fastapi.py
├── app_streamlit.py
├── best_mlp.pth
├── best_multitask_model.pth
├── best_transformer.pth
├── DDLS_Enhanced_MemoryOptimized.ipynb
├── models/
│   ├── mlp_config.pkl
│   ├── multitask_mlp_model.pt
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   ├── transformer_config.pkl
│   └── transformer_model.pt
├── requirements.txt
└── simple_frontend.html
```

*   `DDLS_Enhanced_MemoryOptimized.ipynb`: The main Jupyter notebook containing the data analysis, preprocessing, model training, and evaluation.
*   `app_fastapi.py`: A FastAPI application to serve the trained model as an API.
*   `app_streamlit.py`: A Streamlit web application to interact with the model.
*   `simple_frontend.html`: An HTML/JavaScript frontend for the FastAPI application.
*   `requirements.txt`: A list of Python packages required to run the project.
*   `best_mlp.pth`, `best_multitask_model.pth`, `best_transformer.pth`: Pre-trained model weights.
*   `models/`: A directory containing the model and preprocessor files for the Streamlit application.
*   `Final_Project_Report.md`: The final project report.

## How to Reproduce

1.  **Clone the repository:**
    ```bash
    git clone [URL to your repository]
    cd [repository name]
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Jupyter Notebook:**
    Open and run the `DDLS_Enhanced_MemoryOptimized.ipynb` notebook to perform the data analysis, train the models, and generate the results.

4.  **Run the Web Application:**
    You can choose to run either the FastAPI or the Streamlit application.

    **Option A: FastAPI and HTML Frontend**
    1.  Start the FastAPI server:
        ```bash
        uvicorn app_fastapi:app --reload
        ```
    2.  Open the `simple_frontend.html` file in your web browser.

    **Option B: Streamlit Application**
    1.  Run the Streamlit app:
        ```bash
        streamlit run app_streamlit.py
        ```
    2.  Open the URL provided by Streamlit in your web browser.
