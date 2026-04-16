🧠 Text Classification System

AI/ML Engineer Intern – Technical Assignment

📌 Issued By

Ardentix

🎯 Objective

The objective of this project is to build a machine learning–based text classification system that takes raw text as input and predicts the correct category.
This project demonstrates the complete machine learning pipeline, including data preprocessing, feature extraction, model training, evaluation, and deployment.

📝 Problem Statement

Text data is unstructured and cannot be directly used by machine learning models.
The goal is to preprocess textual data, convert it into numerical form, train supervised learning models, and evaluate their performance on classification tasks.

For this assignment, SMS Spam Detection is chosen as the use case.

📂 Dataset

SMS Spam Collection Dataset

Source: UCI Machine Learning Repository / Kaggle

Total records: ~5,500 SMS messages

Classes:

Spam

Ham (Not Spam)

Why this dataset?

Widely used benchmark dataset for NLP

Suitable for classical ML models like Naive Bayes and Logistic Regression

Clean and well-labeled

🛠️ Technologies & Libraries Used
Category	Tools
Programming Language	Python
Data Handling	NumPy, Pandas
NLP	NLTK
Feature Extraction	TF-IDF
Machine Learning	Scikit-learn
Visualization	Matplotlib, Seaborn
Web Interface	Streamlit
Version Control	Git & GitHub
🔄 Machine Learning Pipeline
1️⃣ Text Preprocessing

The following preprocessing steps are applied:

Convert text to lowercase

Remove punctuation and special characters

Tokenization

Stopword removal using NLTK

2️⃣ Feature Engineering

TF-IDF (Term Frequency–Inverse Document Frequency) is used to convert text into numerical features.

Why TF-IDF?

Reduces the impact of frequently occurring words

Highlights important words

Improves classification performance compared to Bag of Words

3️⃣ Models Used
🔹 Naive Bayes (MultinomialNB)

Fast and efficient for text classification

Works well with word frequency features

🔹 Logistic Regression

Linear classifier with strong decision boundaries

Provides probability-based predictions

Achieved better overall performance

Final Model Selected: Logistic Regression

📊 Model Evaluation
Evaluation Metrics:

Accuracy

Precision

Recall

F1-Score

Results Summary:
Model	Accuracy
Naive Bayes	~97%
Logistic Regression	~98%

Observation:
Logistic Regression slightly outperformed Naive Bayes in accuracy and F1-score.

🖥️ Streamlit Web Application

A simple web interface is built using Streamlit where users can:

Enter SMS text

Get real-time classification (Spam / Ham)

📁 Project Structure
text-classification-ardentix/
│
├── data/
│   └── spam.csv
│
├── app.py
├── train_model.py
├── preprocessing.py
├── requirements.txt
├── model.pkl
├── vectorizer.pkl
└── README.md

⚙️ How to Run the Project
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Train the model
python train_model.py

3️⃣ Run the Streamlit app
streamlit run app.py

🚀 Optional Enhancements Implemented

Model comparison

Clean preprocessing pipeline

Streamlit-based UI

Model persistence using pickle

📌 Conclusion

This project demonstrates a complete end-to-end text classification system, covering data preprocessing, feature extraction, model building, evaluation, and deployment.
It reflects real-world machine learning practices and aligns with the expectations for the AI/ML Engineer Intern role at Ardentix.

🔗 Submission

Code hosted on a public GitHub repository

GitHub link submitted via the official Google Form

🙌 Author

Likhith
AI/ML Engineer Intern Applicant
