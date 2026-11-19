AI, Machine Learning, and Data Science Portfolio
I'm Waeil Mohamadi, an entry-level candidate building projects across full stack data science (FSDS), machine learning, generative AI, and agentic AI. This repository contains runnable applications as well as practice notebooks and course exercises. The projects below are the best starting points.
Featured projects
Project	What it demonstrates	Stack
[Customer Churn Prediction](Customer-Churn-Prediction/)	Reproducible classification pipeline, train-only cross-validation, held-out evaluation, a Gradio interface, and a FastAPI endpoint. The recorded test ROC AUC is 0.8436 on the included sample dataset.	Python, pandas, scikit-learn, Gradio, FastAPI
[Document Q&A Assistant](Document-QA-Assistant/)	PDF question answering with OCR for scanned pages, local embeddings, vector retrieval, conversational follow-ups, and page-linked source passages.	Python, Streamlit, PyMuPDF, RapidOCR, Sentence Transformers, ChromaDB, Groq
[Agentic Shopping Cart Assistant](Agentic-Shopping-Cart-MCP/)	A Spring Boot MCP server that exposes cart and product actions to an external AI assistant, backed by a MySQL cart.	Java 17, Spring Boot, Spring AI, MCP, MySQL, JUnit
EDA and Regression Analysis App	Interactive data exploration, missing-value handling, charts, and four regression models.	Python, Streamlit, pandas, scikit-learn


Each project folder has its own README with setup instructions, scope, and limitations. Other notebooks and scripts in the root are learning exercises and experiments.
Areas I am learning and applying
- Data science and ML: exploratory analysis, cleaning, feature preprocessing, classification, regression, cross-validation, and model evaluation.
- Generative AI: prompt engineering, retrieval-augmented generation (RAG), embeddings, OCR, and source-grounded answers.
- Agentic AI: LLM tool calling and Model Context Protocol (MCP) integrations.
- Application development: Streamlit and Gradio interfaces, FastAPI services, and Spring Boot APIs.
EDA and Regression Analysis App
The root-level Streamlit app lets you upload a CSV or explore an included nuclear energy dataset. It shows data types, summary statistics, missing values, correlation and distribution charts, and pair plots. You can fill missing values, select numeric features and a target, and train linear, polynomial, K-nearest neighbors, or random forest regression.

Run locally:
python -m venv .venv
python -m pip install -r requirements.txt
streamlit run app.py
Use the Python interpreter from your virtual environment for the install and run commands. The app entry point is app.py; src/eda.py contains exploration and cleaning functions, src/model.py contains regression helpers, and archive/ contains example datasets.

Evaluation note: The app currently calculates R² and mean squared error on the same rows used for fitting. These are training-fit metrics, not estimates of performance on unseen data. For a project with a held-out test evaluation, see [Customer Churn Prediction](Customer-Churn-Prediction/).
Contact
LinkedIn · GitHub profile
