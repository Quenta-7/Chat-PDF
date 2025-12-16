import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import rag_core # Importamos nuestra lógica

load_dotenv() # Cargar API KEY

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Procesar con el RAG Core
            num_chunks, num_pages = rag_core.process_document(filepath)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'pages': num_pages,
                'chunks': num_chunks,
                'message': 'Ingesta RAG completada (Hybrid Search activado)'
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
        
    try:
        # Obtener respuesta y citas del Core
        response_text, sources = rag_core.get_answer(query)
        
        return jsonify({
            'response': response_text,
            'sources': sources
        })
    except Exception as e:
        print(e)
        return jsonify({'error': 'Error generating response'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)