# Movie Review Sentiment Intelligence Dashboard

An end-to-end Natural Language Processing (NLP) machine learning application that analyzes sentiment from movie reviews. The system classifies textual reviews as Positive or Negative with 89.24% accuracy using a Logistic Regression model trained on 50,000 IMDB reviews.

It features a full-fledged Flask Web Application with a custom CSS dashboard supporting both Batch Document Analysis CSV and real-time Individual Review Testing.

---

## Features

High Accuracy ML Engine: Trained on 50,000 IMDB movie reviews using TF-IDF Vectorization with bigrams (`ngram_range=(1,2)`) and negation-preserving preprocessing.
Multi-Format Batch File Processing: Upload datasets in CSV formats for bulk sentiment extraction.
Visual Analytics: Automatically generates sentiment distribution metrics (Total Count, Positive %, Negative %) and a Matplotlib pie chart visualization.
Single Review Tester: Instant prediction tool for testing individual sentences or custom reviews.
Custom Modern Dashboard: Sleek, responsive tabbed UI built using pure CSS .

---

## Tech Stack & Libraries

Language: Python 3.13
Machine Learning & NLP: Scikit-Learn, NLTK, Pandas, Joblib
Web Framework: Flask
Data Visualization: Matplotlib
Frontend: HTML5, Custom CSS, JavaScript

---

## Model Performance Comparison

During development, two models were trained and benchmarked on the test set (9,917 samples):

Model | Accuracy | F1-Score (Negative) | F1-Score (Positive) |

Logistic Regression (Selected) | 89.24% | 0.89 | 0.89 |
Multinomial Naive Bayes | 86.82% | 0.87 | 0.87 |

---

## Project Structure

```text
sentiment_flask_app/
│
├── sentiment_model.pkl       # Saved Logistic Regression Model
├── tfidf_vectorizer.pkl      # Saved TF-IDF Vectorizer
├── main.py                    # Main Flask Backend Logic
├── static/
│   └── style.css            # Custom Styling File
└── templates/
    └── index.html           # Dashboard Interface



Getting Started

1. Clone the Repository

git clone (https://github.com/abhisheknishad23/Sentiment-Analysis.git)
cd Sentiment-Analysis

2. Create and Activate Virtual Environment

# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies

pip install flask pandas scikit-learn joblib matplotlib nltk

4. Run the Flask Application

python app.py
Open your browser and navigate to http://127.0.0.1:5000/