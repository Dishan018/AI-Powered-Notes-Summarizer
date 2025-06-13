# 🧠 AI-Powered Notes Summarizer

A simple yet powerful web app that summarizes long notes, articles, or PDFs using Natural Language Processing (NLP). Built with Python, Flask, and Hugging Face Transformers, this tool helps students and professionals quickly generate clean summaries from large documents.

---

## 🚀 Features

- 📄 Upload text or PDF files
- 🤖 AI-generated summaries using Hugging Face models
- 🌐 Simple web interface (Flask + HTML/CSS)
- 💾 Local processing — no cloud login required
- ⚡ Fast, accurate, and responsive

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask  
- **AI/NLP**: Hugging Face Transformers (BART model)  
- **Frontend**: HTML, CSS  
- **PDF Parsing**: PyPDF2

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/your-username/ai-notes-summarizer.git
cd ai-notes-summarizer

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
