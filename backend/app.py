import os
import pathlib
from flask import Flask, request, jsonify, send_from_directory # Added send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import google.generativeai as genai
from PIL import Image
import PyPDF2
from docx import Document
import time

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env')) # Ensure .env in backend/ is loaded

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")
genai.configure(api_key=GEMINI_API_KEY)

# --- Define path to the frontend directory ---
# Assuming your app.py is in 'backend/', and 'frontend/' is a sibling directory
# So, from 'backend/', we go up one level ('../') and then into 'frontend/'
FRONTEND_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'frontend')

# Initialize Flask app
# static_folder is set to your frontend folder
# static_url_path='' means files in static_folder will be served from the root URL
# e.g., frontend/index.html will be accessible at /index.html (or just / if it's the default)
app = Flask(__name__, static_folder=FRONTEND_FOLDER, static_url_path='')
CORS(app) # Enable CORS for all routes

# Configure upload folder and allowed extensions (for backend processing)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads') # Place uploads within backend for clarity
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ... (rest of your helper functions: allowed_file, extract_text_from_pdf, etc. - NO CHANGES NEEDED THERE) ...
# --- Configuration for large file handling ---
MAX_CHARS_FOR_FULL_CONTEXT = 300000
CHUNK_SIZE_CHARS = 25000
CHUNK_OVERLAP_CHARS = 2000

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page_text = reader.pages[page_num].extract_text()
                if page_text:
                    text += page_text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
    return text

def extract_text_from_docx(file_path):
    text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error extracting DOCX text: {e}")
    return text

def summarize_text_with_gemini(text_to_summarize, model_name='gemini-1.5-flash',
                               instruction="Provide a concise and comprehensive summary of the following text segment:"):
    if not text_to_summarize.strip():
        return ""
    try:
        summarization_model = genai.GenerativeModel(model_name)
        prompt = f"{instruction}\n\n---\n{text_to_summarize}\n---"
        if len(prompt) > 3500000:
             print(f"Warning: Text for a single summarization call is very long ({len(prompt)} chars). Truncating for safety.")
             prompt = prompt[:3500000]
        response = summarization_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error during summarization call to Gemini: {e}")
        return f"[Error summarizing. Original text segment starts with: {text_to_summarize[:300]}...]"

def chunk_text(text, chunk_size=CHUNK_SIZE_CHARS, overlap=CHUNK_OVERLAP_CHARS):
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        if end == text_len:
            break
        start += (chunk_size - overlap)
        if start >= text_len:
            break
    return chunks


# --- API Endpoint ---
@app.route('/chat', methods=['POST'])
def chat():
    # ... (your existing /chat logic - NO CHANGES NEEDED HERE) ...
    try:
        question = request.form.get('question')
        file = request.files.get('file')

        if not question:
            if not file:
                return jsonify({"error": "No question or file provided"}), 400
            filename_for_default_q = secure_filename(file.filename) if file and file.filename else "the uploaded file"
            question = f"Please analyze the content of '{filename_for_default_q}'. Provide a summary or answer any implicit questions based on its content."

        processed_file_content_for_llm = None
        context_message_for_prompt = ""

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_extension = filename.rsplit('.', 1)[1].lower()
            context_message_for_prompt = f"\n\n--- Regarding User Uploaded File: {filename} ---\n"
            if file_extension in {'png', 'jpg', 'jpeg', 'gif'}:
                img = Image.open(file_path)
                processed_file_content_for_llm = img
                context_message_for_prompt += "[Content of this image is provided directly to the model.]"
            else:
                extracted_text = ""
                if file_extension == 'pdf':
                    extracted_text = extract_text_from_pdf(file_path)
                elif file_extension == 'docx':
                    extracted_text = extract_text_from_docx(file_path)
                elif file_extension == 'txt':
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f: extracted_text = f.read()
                    except UnicodeDecodeError:
                        with open(file_path, 'r', encoding='latin-1') as f: extracted_text = f.read()
                if not extracted_text.strip():
                    context_message_for_prompt += "[File appears to be empty or text could not be extracted.]"
                    processed_file_content_for_llm = ""
                elif len(extracted_text) > MAX_CHARS_FOR_FULL_CONTEXT:
                    context_message_for_prompt += f"[File content is extensive ({len(extracted_text):,} chars). A summary is being provided to the model.]\n"
                    text_chunks = chunk_text(extracted_text)
                    chunk_summaries = [summarize_text_with_gemini(chunk, instruction="Concisely summarize this text segment...") for chunk in text_chunks]
                    combined_summaries = "\n\n---\n\n".join(chunk_summaries)
                    if len(combined_summaries) > MAX_CHARS_FOR_FULL_CONTEXT * 0.75:
                        final_summary = summarize_text_with_gemini(combined_summaries, instruction="Synthesize these summaries into one cohesive summary:")
                        processed_file_content_for_llm = final_summary
                        context_message_for_prompt += f"**Final Synthesized Summary of File Content:**\n"
                    else:
                        processed_file_content_for_llm = combined_summaries
                        context_message_for_prompt += f"**Combined Summaries of File Content:**\n"
                else:
                    processed_file_content_for_llm = extracted_text
                    context_message_for_prompt += f"[The full text content of the file (or its beginning) is provided below.]\n"
            try: os.remove(file_path)
            except OSError as e: print(f"Error removing uploaded file {file_path}: {e}")

        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt_parts = [question]
        if context_message_for_prompt: prompt_parts.append(context_message_for_prompt)
        if processed_file_content_for_llm: prompt_parts.append(processed_file_content_for_llm)
        if file: prompt_parts.append("\n--- End of file context ---\nPlease answer the initial question...")
        if not any(prompt_parts) or (len(prompt_parts) == 1 and not str(prompt_parts[0]).strip()):
             return jsonify({"answer": "Issue processing request or file. Try again or rephrase."})
        response = model.generate_content(prompt_parts)
        return jsonify({"answer": response.text})
    except Exception as e:
        import traceback
        print(f"Critical Error in /chat endpoint: {e}")
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


# --- Route to serve the main frontend (chatbot) HTML file ---
@app.route('/') # Serves index.html from the root URL
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

# --- Route to serve any other static files from the frontend folder (CSS, JS, images if any) ---
# This is often handled automatically by Flask if static_url_path is set correctly,
# but explicitly adding it can be clearer or useful for specific files.
# If your index.html doesn't reference other files from the frontend folder, you might not strictly need this.
@app.route('/<path:filename>')
def serve_static_files(filename):
    return send_from_directory(app.static_folder, filename)


if __name__ == '__main__':
    # This is for local development only. Gunicorn will be used on Render.
    app.run(debug=True, port=5000)
