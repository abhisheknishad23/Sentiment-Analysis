from flask import Flask, render_template, request
import joblib
import pandas as pd
import re
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

@app.route('/')
def home():
    return render_template('index.html')

# 1. Batch Analysis Route
@app.route('/analyze_csv', methods=['POST'])
def analyze_csv():
    if 'file' not in request.files or request.files['file'].filename == '':
        return render_template('index.html', error="Please upload a valid CSV file.")

    df = pd.read_csv(request.files['file'])
    text_col = 'review' if 'review' in df.columns else df.columns[0]
    
    df['clean'] = df[text_col].apply(clean_text)
    vectors = vectorizer.transform(df['clean'])
    df['prediction'] = model.predict(vectors)
    
    total = len(df)
    pos = int((df['prediction'] == 1).sum())
    neg = int((df['prediction'] == 0).sum())
    
    # Generate Pie Chart
    plt.figure(figsize=(4, 4))
    plt.pie([pos, neg], labels=['Positive', 'Negative'], colors=['#4CAF50', '#FF5252'], autopct='%1.1f%%', startangle=140)
    plt.title('Overall Sentiment Distribution')
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return render_template('index.html', total=total, pos=pos, neg=neg, 
                           pos_pct=round(pos/total*100, 2), neg_pct=round(neg/total*100, 2), 
                           chart=chart_url)

# 2. Individual Review Check Route
@app.route('/predict_single', methods=['POST'])
def predict_single():
    single_review = request.form.get('single_review', '')
    if not single_review.strip():
        return render_template('index.html', single_error="Please enter text to test.")
    
    cleaned = clean_text(single_review)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    result = "Positive 😊" if pred == 1 else "Negative 😞"
    
    return render_template('index.html', single_review=single_review, single_result=result)

if __name__ == '__main__':
    app.run(debug=True)