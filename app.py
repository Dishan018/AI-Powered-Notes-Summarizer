# app.py
from flask import Flask, render_template, request
from transformers import pipeline
import PyPDF2

app = Flask(__name__)

# Load the summarization pipeline (you can use T5 or BART as well)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    if file.filename.endswith('.pdf'):
        text = extract_text_from_pdf(file)
    else:
        text = file.read().decode('utf-8')

    # Limit text length to avoid model input overflow
    text = text[:4000]

    # Generate summary
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']

    return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
