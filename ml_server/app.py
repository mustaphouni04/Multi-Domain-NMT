
import os
import sys
import json
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from transformers import MarianMTModel, MarianTokenizer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='../static', static_url_path='')
CORS(app)

# Global variables for models
models = {}
current_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_name):
    """Load a translation model"""
    try:
        if model_name in models:
            return models[model_name]
        
        logger.info(f"Loading model: {model_name}")
        
        # Check if it's a local model path or HuggingFace model
        if os.path.exists(f"domain_models/{model_name}"):
            model_path = f"domain_models/{model_name}"
        else:
            model_path = model_name
        
        tokenizer = MarianTokenizer.from_pretrained(model_path)
        model = MarianMTModel.from_pretrained(model_path).to(device)
        
        models[model_name] = {
            'model': model,
            'tokenizer': tokenizer,
            'name': model_name
        }
        
        logger.info(f"Successfully loaded model: {model_name}")
        return models[model_name]
        
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        raise e

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': device,
        'models_loaded': list(models.keys())
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models"""
    try:
        available_models = []
        
        # Check for local domain models
        if os.path.exists('domain_models'):
            for model_dir in os.listdir('domain_models'):
                if os.path.isdir(f'domain_models/{model_dir}'):
                    available_models.append({
                        'id': model_dir,
                        'name': model_dir.replace('_', ' ').title(),
                        'type': 'domain',
                        'loaded': model_dir in models
                    })
        
        # Add default student model
        default_model = "Helsinki-NLP/opus-mt-en-es"
        available_models.append({
            'id': default_model,
            'name': 'Helsinki English-Spanish',
            'type': 'general',
            'loaded': default_model in models
        })
        
        return jsonify({'models': available_models})
        
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/load', methods=['POST'])
def load_model_endpoint():
    """Load a specific model"""
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        
        if not model_name:
            return jsonify({'error': 'Model name is required'}), 400
        
        model_info = load_model(model_name)
        global current_model
        current_model = model_name
        
        return jsonify({
            'message': f'Model {model_name} loaded successfully',
            'current_model': current_model
        })
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/translate', methods=['POST'])
def translate():
    """Translate text"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        model_name = data.get('model', current_model)
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        if not model_name:
            # Use default model
            model_name = "Helsinki-NLP/opus-mt-en-es"
        
        # Load model if not already loaded
        model_info = load_model(model_name)
        
        # Perform translation
        tokenizer = model_info['tokenizer']
        model = model_info['model']
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate translation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
        
        # Decode translation
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return jsonify({
            'translation': translation,
            'model_used': model_name,
            'input_text': text
        })
        
    except Exception as e:
        logger.error(f"Error during translation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start model training"""
    try:
        data = request.get_json()
        training_type = data.get('type', 'maml')
        
        # This is a placeholder - you would integrate with your actual training scripts
        logger.info(f"Starting {training_type} training...")
        
        return jsonify({
            'message': f'{training_type.upper()} training started',
            'status': 'started',
            'training_type': training_type
        })
        
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/status', methods=['GET'])
def get_training_status():
    """Get training status"""
    try:
        # This is a placeholder - you would check actual training status
        return jsonify({
            'status': 'idle',
            'progress': 0,
            'current_epoch': 0,
            'total_epochs': 0
        })
        
    except Exception as e:
        logger.error(f"Error getting training status: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Serve React frontend
@app.route('/')
def serve_frontend():
    """Serve the React frontend index.html"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def catch_all(path):
    """Catch-all route for React Router and static files"""
    try:
        # Check if it's a static file that exists
        if os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        else:
            # For React Router routes, serve index.html
            return send_from_directory(app.static_folder, 'index.html')
    except Exception as e:
        logger.error(f"Error serving file {path}: {str(e)}")
        # Fallback to index.html for any errors
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    logger.info(f"Using device: {device}")
    logger.info(f"Static folder: {app.static_folder}")
    
    # Load default model
    try:
        default_model = "Helsinki-NLP/opus-mt-en-es"
        load_model(default_model)
        current_model = default_model
        logger.info(f"Default model loaded: {default_model}")
    except Exception as e:
        logger.warning(f"Could not load default model: {str(e)}")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
