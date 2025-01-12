import os
from flask import Flask, render_template, request, send_from_directory
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import string
import PyPDF2

app = Flask(__name__)

# Pastikan lokasi folder ini sesuai dengan struktur proyek Anda
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, 'data2')


def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return word_tokenize(text)


def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() or ''  # Tambahkan konten setiap halaman
            return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ''


def prepare_bm25():
    files = os.listdir(PDF_FOLDER)
    documents = []
    file_names = []

    for file_name in files:
        if file_name.endswith('.pdf'):
            pdf_path = os.path.join(PDF_FOLDER, file_name)
            text = extract_text_from_pdf(pdf_path)
            if text:
                documents.append(clean_text(text))
                file_names.append(file_name)

    bm25 = BM25Okapi(documents)
    return bm25, file_names, documents


@app.route('/data2/<path:filename>')
def serve_pdf(filename):
    # Pastikan Flask mengirimkan PDF dengan header yang benar
    return send_from_directory(
        PDF_FOLDER,
        filename,
        as_attachment=False,  # Menampilkan langsung di browser
        mimetype='application/pdf'
    )



@app.route('/', methods=['GET', 'POST'])
def index():
    bm25, file_names, documents = prepare_bm25()
    results = []

    if request.method == 'POST':
        query = request.form['query']
        query_tokens = clean_text(query)
        scores = bm25.get_scores(query_tokens)

        for idx, score in enumerate(scores):
            if score > 0:
                results.append({
                    'file_name': file_names[idx],
                    'score': score,
                    'file_path': f'/data2/{file_names[idx]}'
                })

        results = sorted(results, key=lambda x: x['score'], reverse=True)

    return render_template('index.html', results=results)


if __name__ == '__main__':
    app.run(debug=True)