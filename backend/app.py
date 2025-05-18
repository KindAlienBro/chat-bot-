import os
import pathlib
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import google.generativeai as genai
from PIL import Image
import PyPDF2
from docx import Document
import time # For potential delays if needed, not strictly for Gemini usually

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Flask app
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- Configuration for large file handling ---
# Gemini 1.5 Flash/Pro have large context windows (1M tokens ~ 4M chars).
# These limits are for demonstration of chunking/summarization logic.
# Adjust based on your needs, model choice, and cost considerations.
MAX_CHARS_FOR_FULL_CONTEXT = 300000  # Approx 75k tokens. If text is less, send as is.
# If text is > MAX_CHARS_FOR_FULL_CONTEXT, it will be summarized.
# For summarization of very large texts, we chunk it first:
CHUNK_SIZE_CHARS = 25000  # Size of chunks for summarization pass
CHUNK_OVERLAP_CHARS = 2000 # Overlap between chunks to maintain context

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

# --- Helper function to summarize text using Gemini ---
def summarize_text_with_gemini(text_to_summarize, model_name='gemini-1.5-flash', 
                               instruction="Provide a concise and comprehensive summary of the following text segment:"):
    if not text_to_summarize.strip():
        return ""
    try:
        # Using 1.5 Flash for summarization as it's fast and capable
        summarization_model = genai.GenerativeModel(model_name)
        prompt = f"{instruction}\n\n---\n{text_to_summarize}\n---"
        
        # Gemini models handle large inputs, but be mindful of overall complexity and cost.
        # This check is a safeguard for extremely large individual summarization requests.
        if len(prompt) > 3500000: # A very large single prompt
             print(f"Warning: Text for a single summarization call is very long ({len(prompt)} chars). Truncating for safety.")
             prompt = prompt[:3500000]

        response = summarization_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error during summarization call to Gemini: {e}")
        # Fallback: return a truncated version or an error message
        return f"[Error summarizing. Original text segment starts with: {text_to_summarize[:300]}...]"

# --- Helper function for chunking text ---
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
        # Ensure start doesn't go way past due to large overlap on small final piece
        if start >= text_len: # Should not happen if end == text_len breaks
            break 
    return chunks

@app.route('/chat', methods=['POST'])
def chat():
    try:
        question = request.form.get('question')
        file = request.files.get('file')

        # Handle default question if only a file is uploaded
        if not question:
            if not file:
                return jsonify({"error": "No question or file provided"}), 400
            # If file is present, but no question, formulate a default question.
            filename_for_default_q = secure_filename(file.filename) if file and file.filename else "the uploaded file"
            question = f"Please analyze the content of '{filename_for_default_q}'. Provide a summary or answer any implicit questions based on its content."

        processed_file_content_for_llm = None # This will hold text (original/summary) or PIL Image
        context_message_for_prompt = "" # A message to prepend to the LLM about the file

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_extension = filename.rsplit('.', 1)[1].lower()

            context_message_for_prompt = f"\n\n--- Regarding User Uploaded File: {filename} ---\n"

            if file_extension in {'png', 'jpg', 'jpeg', 'gif'}:
                img = Image.open(file_path)
                processed_file_content_for_llm = img # Store PIL image
                context_message_for_prompt += "[Content of this image is provided directly to the model.]"
            
            else: # Text-based files (txt, pdf, docx)
                extracted_text = ""
                if file_extension == 'pdf':
                    extracted_text = extract_text_from_pdf(file_path)
                elif file_extension == 'docx':
                    extracted_text = extract_text_from_docx(file_path)
                elif file_extension == 'txt':
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            extracted_text = f.read()
                    except UnicodeDecodeError: # Try another common encoding as fallback
                        with open(file_path, 'r', encoding='latin-1') as f:
                            extracted_text = f.read()
                
                if not extracted_text.strip():
                    context_message_for_prompt += "[File appears to be empty or text could not be extracted.]"
                    processed_file_content_for_llm = "" # Ensure it's an empty string
                elif len(extracted_text) > MAX_CHARS_FOR_FULL_CONTEXT:
                    print(f"File '{filename}' is large ({len(extracted_text):,} chars). Applying summarization strategy.")
                    context_message_for_prompt += f"[File content is extensive ({len(extracted_text):,} chars). A summary is being provided to the model.]\n"
                    
                    text_chunks = chunk_text(extracted_text, chunk_size=CHUNK_SIZE_CHARS, overlap=CHUNK_OVERLAP_CHARS)
                    chunk_summaries = []
                    
                    print(f"Processing {len(text_chunks)} chunks for summarization...")
                    for i, chunk in enumerate(text_chunks):
                        print(f"Summarizing chunk {i+1}/{len(text_chunks)} ({len(chunk):,} chars)...")
                        # Optional: add a small delay if you anticipate hitting strict, very low rate limits
                        # time.sleep(0.5) # Usually not needed for Gemini API's typical limits
                        chunk_summary = summarize_text_with_gemini(chunk, instruction="Concisely summarize this text segment, capturing key information and context:")
                        chunk_summaries.append(chunk_summary)
                        print(f"Chunk {i+1} summary length: {len(chunk_summary):,} chars")
                    
                    combined_summaries = "\n\n---\n\n".join(chunk_summaries) # Join summaries with a clear separator
                    
                    # If combined summaries are still very long, do a final summary pass
                    if len(combined_summaries) > MAX_CHARS_FOR_FULL_CONTEXT * 0.75: # Heuristic
                        print(f"Combined summaries are still long ({len(combined_summaries):,} chars). Performing a final summary.")
                        final_summary = summarize_text_with_gemini(combined_summaries, instruction="Synthesize the following collection of summaries into one cohesive and comprehensive summary:")
                        processed_file_content_for_llm = final_summary
                        context_message_for_prompt += f"**Final Synthesized Summary of File Content:**\n" # (Actual summary text will be appended later)
                        print(f"Final summary length: {len(final_summary):,} chars")
                    else:
                        processed_file_content_for_llm = combined_summaries
                        context_message_for_prompt += f"**Combined Summaries of File Content:**\n" # (Actual summary text will be appended later)
                else:
                    # File is within size limits for direct inclusion
                    processed_file_content_for_llm = extracted_text
                    context_message_for_prompt += f"[The full text content of the file (or its beginning) is provided below.]\n"
            
            # Clean up uploaded file after processing (optional, but good practice)
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error removing uploaded file {file_path}: {e}")


        # --- Gemini API Call ---
        # Using gemini-1.5-flash as it's fast, cost-effective, and has a large context window.
        # You can switch to 'gemini-1.5-pro' for potentially higher quality on complex tasks.
        model = genai.GenerativeModel('gemini-1.5-flash') 
        
        prompt_parts = [question] # Start with the user's question

        if context_message_for_prompt: # Add the context message about the file
             prompt_parts.append(context_message_for_prompt)

        # Add the actual processed file content (PIL Image or text/summary)
        if processed_file_content_for_llm: # Could be PIL.Image or string
            prompt_parts.append(processed_file_content_for_llm) 
        
        # Add a concluding instruction if file content was involved
        if file:
            prompt_parts.append("\n--- End of file context ---\nPlease answer the initial question, considering the above file content if relevant.")

        # For debugging the prompt structure:
        # print("---- FINAL PROMPT PARTS TO GEMINI ----")
        # for i, part in enumerate(prompt_parts):
        #     if isinstance(part, str):
        #         print(f"Part {i} (text, len {len(part)}): {part[:300].replace(chr(10), ' ')}...") # Limit print, replace newlines
        #     elif isinstance(part, Image.Image):
        #         print(f"Part {i} (image): {part.format} {part.size}")
        # print("--------------------------------------")

        # Ensure there's something to send
        if not any(prompt_parts) or (len(prompt_parts) == 1 and not str(prompt_parts[0]).strip()):
             return jsonify({"answer": "It seems there was an issue processing your request or the file. Could you try again or rephrase?"})

        response = model.generate_content(prompt_parts)
        
        # print(f"DEBUG: Gemini response raw: {response}") # For debugging
        # print(f"DEBUG: Gemini response text: {response.text}") # For debugging

        return jsonify({"answer": response.text})

    except Exception as e:
        import traceback
        print(f"Critical Error in /chat endpoint: {e}")
        traceback.print_exc() # This will print the full traceback to your Flask console
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)