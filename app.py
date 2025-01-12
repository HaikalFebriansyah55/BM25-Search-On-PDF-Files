import os
from flask import Flask, render_template, request, send_from_directory
from rank_bm25 import BM25Okapi
import string
import PyPDF2
import re

app = Flask(__name__)

# Lokasi folder PDF
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, 'data_pdf')

# Fungsi untuk highlight teks yang dicari
def highlight_text(text, query):
    query_words = set(word.lower() for word in query.split())
    words = re.findall(r'\b\w+\b', text)
    highlighted_words = [
        f"<mark>{word}</mark>" if word.lower() in query_words else word for word in words
    ]
    return ' '.join(highlighted_words)

# Fungsi untuk membersihkan teks
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Menghapus karakter selain huruf dan angka
    return text.split()

# Fungsi untuk membaca teks dari file PDF
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''.join(page.extract_text() or '' for page in reader.pages)
            return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

# Fungsi untuk membuat snippet (memotong 3-4 kata di awal teks)
def create_snippet(text, query, num_words=4):
    words = text.split()
    snippet = ' '.join(words[:num_words])  # Memotong 3-4 kata di awal teks
    highlighted_snippet = highlight_text(snippet, query)  # Highlight kata yang sesuai query
    return highlighted_snippet

# Fungsi untuk menyiapkan BM25
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

# Route untuk melihat isi PDF dengan highlight
@app.route('/view_pdf/<filename>')
def view_pdf(filename):
    query = request.args.get('query', '')  # Mendapatkan kata kunci pencarian
    pdf_path = os.path.join(PDF_FOLDER, filename)
    text = extract_text_from_pdf(pdf_path)

    if text:
        if query:
            highlighted_text = highlight_text(text, query)
        else:
            highlighted_text = f'<pre>{text}</pre>'
        return render_template('view_pdf.html', content=highlighted_text, filename=filename)
    return "Error: Could not read the PDF file."

# Route untuk mengunduh PDF
@app.route('/download_pdf/<filename>')
def download_pdf(filename):
    return send_from_directory(PDF_FOLDER, filename, as_attachment=True)

# Route utama untuk pencarian
@app.route('/', methods=['GET', 'POST'])
def index():
    bm25, file_names, documents = prepare_bm25()
    results = []

    if request.method == 'POST':
        query = request.form['query']
        if not query:
            return render_template('index.html', results=[], error="Please enter a query.")  # Validasi input kosong
        query_tokens = clean_text(query)
        scores = bm25.get_scores(query_tokens)

        for idx, score in enumerate(scores):
            if score > 0:
                # Ambil teks asli dokumen
                original_text = ' '.join(documents[idx])  # Gabungkan teks yang sudah di-tokenize
                # Buat snippet
                snippet = create_snippet(original_text, query)
                results.append({
                    'file_name': file_names[idx],
                    'score': f"{score:.2f}",
                    'snippet': snippet,
                    'file_path': f'/view_pdf/{file_names[idx]}?query={query}'
                })

        results = sorted(results, key=lambda x: x['score'], reverse=True)

    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5000)
