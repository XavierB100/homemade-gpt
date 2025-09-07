#!/usr/bin/env python3
"""
Enhanced HomeMade GPT Web Interface
A Flask web application for training and chatting with custom GPT models
"""

import os
import json
import threading
import time
from datetime import datetime
from pathlib import Path
import torch
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

# Import our HomeMade GPT modules
from data_loader import DataProcessor
from enhanced_gpt import GPT, GPTConfig
from chat import ChatBot

app = Flask(__name__)
app.config['SECRET_KEY'] = 'homemade-gpt-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
Path('uploads').mkdir(exist_ok=True)
Path('models').mkdir(exist_ok=True)
Path('logs').mkdir(exist_ok=True)

# Initialize SocketIO for real-time updates
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables for training status
training_status = {
    'is_training': False,
    'progress': 0,
    'current_loss': 0.0,
    'best_loss': float('inf'),
    'iteration': 0,
    'max_iterations': 0,
    'model_name': '',
    'log_messages': []
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'txt', 'log'}

def get_available_models():
    """Get list of available trained models"""
    models = []
    models_dir = Path('models')
    
    for model_file in models_dir.glob('*.pt'):
        if model_file.name in ['ckpt.pt', 'final_model.pt']:
            continue
            
        try:
            # Try to load model info
            checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
            
            model_info = {
                'name': model_file.stem,
                'file': model_file.name,
                'path': str(model_file),
                'size': model_file.stat().st_size,
                'modified': datetime.fromtimestamp(model_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M'),
                'params': 'Unknown',
                'vocab_size': 'Unknown'
            }
            
            if 'model_args' in checkpoint:
                config = checkpoint['model_args']
                # Calculate approximate parameters
                params = (config.vocab_size * config.n_embd + 
                         config.block_size * config.n_embd +
                         config.n_layer * (3 * config.n_embd * config.n_embd + 4 * config.n_embd * config.n_embd))
                model_info['params'] = f"{params/1e6:.1f}M"
                model_info['vocab_size'] = str(config.vocab_size)
            
            models.append(model_info)
            
        except Exception as e:
            # If we can't load the model, still list it but with limited info
            models.append({
                'name': model_file.stem,
                'file': model_file.name,
                'path': str(model_file),
                'size': model_file.stat().st_size,
                'modified': datetime.fromtimestamp(model_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M'),
                'params': 'Error',
                'vocab_size': 'Error',
                'error': str(e)
            })
    
    # Also check for default checkpoints
    for default_model in ['ckpt.pt', 'final_model.pt']:
        model_path = models_dir / default_model
        if model_path.exists():
            models.append({
                'name': f'Latest {default_model.split(".")[0].title()}',
                'file': default_model,
                'path': str(model_path),
                'size': model_path.stat().st_size,
                'modified': datetime.fromtimestamp(model_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M'),
                'params': 'Unknown',
                'vocab_size': 'Unknown'
            })
    
    return sorted(models, key=lambda x: x['modified'], reverse=True)

@app.route('/')
def index():
    """Main homepage"""
    models = get_available_models()
    return render_template('index.html', models=models)

@app.route('/train')
def train_page():
    """Training interface page"""
    return render_template('train.html')

@app.route('/chat')
def chat_page():
    """Chat interface page"""
    models = get_available_models()
    return render_template('chat.html', models=models)

@app.route('/models')
def models_page():
    """Model management page"""
    models = get_available_models()
    return render_template('models.html', models=models)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload for training"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file selected'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Add timestamp to filename to avoid conflicts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{timestamp}{ext}"
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Analyze the file
        try:
            processor = DataProcessor()
            text, metadata = processor.load_and_process_data(file_path, 'auto')
            
            return jsonify({
                'success': True, 
                'filename': filename,
                'file_path': file_path,
                'size': len(text),
                'metadata': metadata,
                'message': f'File uploaded successfully! Detected format: {metadata["format"]}'
            })
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error analyzing file: {str(e)}'})
    
    return jsonify({'success': False, 'message': 'Invalid file type. Please upload .txt files.'})

def run_training(config):
    """Run training in a separate thread"""
    global training_status
    
    try:
        training_status.update({
            'is_training': True,
            'progress': 0,
            'iteration': 0,
            'max_iterations': config['max_iters'],
            'model_name': config['model_name'],
            'log_messages': [],
            'current_loss': 0.0,
            'best_loss': float('inf')
        })
        
        # Import training modules
        import train
        
        # Load and process data
        processor = DataProcessor()
        text, metadata = processor.load_and_process_data(config['data_path'], config['data_type'])
        
        # Build vocabulary
        processor.build_vocabulary(text)
        train_data, val_data = processor.get_train_val_split(text)
        
        # Configure model
        vocab_size = len(processor.chars)
        model_config = GPTConfig.get_preset(config['model_size'], vocab_size, config['block_size'])
        model_config.dropout = config['dropout']
        
        # Create model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = GPT(model_config)
        model.to(device)
        
        # Training configuration
        train_config = {
            'learning_rate': config['learning_rate'],
            'max_iters': config['max_iters'],
            'warmup_iters': min(100, config['max_iters'] // 10),
            'lr_decay_iters': config['max_iters'],
            'min_lr': config['learning_rate'] / 10,
            'beta1': 0.9,
            'beta2': 0.95,
            'grad_clip': 1.0,
            'weight_decay': 1e-1,
        }
        
        # Initialize optimizer
        optimizer = model.configure_optimizers(
            train_config['weight_decay'], 
            train_config['learning_rate'], 
            (train_config['beta1'], train_config['beta2']), 
            device.split(':')[0]
        )
        
        # Training loop with progress updates
        best_val_loss = float('inf')
        
        for iter_num in range(config['max_iters']):
            # Update progress
            progress = int((iter_num / config['max_iters']) * 100)
            training_status.update({
                'progress': progress,
                'iteration': iter_num
            })
            
            # Learning rate scheduling
            lr = train.get_lr(iter_num, train_config)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Forward pass
            X, Y = train.get_batch(train_data, config['batch_size'], config['block_size'], device)
            logits, loss = model(X, Y)
            
            # Backward pass
            loss.backward()
            
            if train_config['grad_clip'] != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config['grad_clip'])
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # Update loss
            current_loss = loss.item()
            training_status['current_loss'] = current_loss
            
            # Periodic evaluation
            if iter_num % (config['max_iters'] // 10) == 0 or iter_num == config['max_iters'] - 1:
                model.eval()
                val_loss = train.estimate_loss(
                    model, train_data, val_data, 50, 
                    config['batch_size'], config['block_size'], 
                    device, torch.no_grad()
                )['val']
                model.train()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    training_status['best_loss'] = best_val_loss
                    
                    # Save best model
                    model_path = f"models/{config['model_name']}.pt"
                    checkpoint = {
                        'model': model.state_dict(),
                        'model_args': model_config,
                        'train_config': train_config,
                        'vocab_size': vocab_size,
                        'processor': {
                            'chars': processor.chars,
                            'stoi': processor.stoi,
                            'itos': processor.itos
                        }
                    }
                    torch.save(checkpoint, model_path)
                
                # Emit progress update
                socketio.emit('training_progress', {
                    'progress': progress,
                    'iteration': iter_num,
                    'current_loss': current_loss,
                    'val_loss': val_loss,
                    'best_loss': best_val_loss,
                    'lr': lr
                })
        
        # Training completed
        training_status.update({
            'is_training': False,
            'progress': 100
        })
        
        socketio.emit('training_complete', {
            'model_name': config['model_name'],
            'final_loss': training_status['best_loss']
        })
        
    except Exception as e:
        training_status.update({
            'is_training': False,
            'progress': 0
        })
        
        socketio.emit('training_error', {
            'error': str(e)
        })

@app.route('/start_training', methods=['POST'])
def start_training():
    """Start model training"""
    global training_status
    
    if training_status['is_training']:
        return jsonify({'success': False, 'message': 'Training already in progress'})
    
    data = request.json
    
    # Validate required fields
    required_fields = ['file_path', 'model_name', 'model_size', 'max_iters']
    for field in required_fields:
        if field not in data:
            return jsonify({'success': False, 'message': f'Missing field: {field}'})
    
    # Prepare training configuration
    config = {
        'data_path': data['file_path'],
        'data_type': data.get('data_type', 'auto'),
        'model_name': data['model_name'],
        'model_size': data['model_size'],
        'block_size': int(data.get('block_size', 256)),
        'batch_size': int(data.get('batch_size', 32)),
        'max_iters': int(data['max_iters']),
        'learning_rate': float(data.get('learning_rate', 3e-4)),
        'dropout': float(data.get('dropout', 0.1))
    }
    
    # Start training in background thread
    training_thread = threading.Thread(target=run_training, args=(config,))
    training_thread.daemon = True
    training_thread.start()
    
    return jsonify({'success': True, 'message': 'Training started!'})

@app.route('/training_status')
def get_training_status():
    """Get current training status"""
    return jsonify(training_status)

@app.route('/chat_api', methods=['POST'])
def chat_api():
    """API endpoint for chat"""
    data = request.json
    
    try:
        model_path = data['model_path']
        message = data['message']
        temperature = float(data.get('temperature', 0.8))
        max_length = int(data.get('max_length', 200))
        
        # Load chatbot (cache this in production)
        chatbot = ChatBot(model_path)
        
        # Generate response
        response = chatbot.generate_response(
            message,
            max_length=max_length,
            temperature=temperature,
            top_k=50,
            top_p=0.9
        )
        
        return jsonify({
            'success': True,
            'response': response.strip()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/delete_model/<model_name>')
def delete_model(model_name):
    """Delete a trained model"""
    try:
        model_path = Path('models') / f"{model_name}"
        if model_path.exists():
            model_path.unlink()
            flash(f'Model {model_name} deleted successfully', 'success')
        else:
            flash(f'Model {model_name} not found', 'error')
    except Exception as e:
        flash(f'Error deleting model: {str(e)}', 'error')
    
    return redirect(url_for('models_page'))

# SocketIO event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connected', {'data': 'Connected to HomeMade GPT server'})

@socketio.on('request_training_status')
def handle_status_request():
    """Send current training status to client"""
    emit('training_status', training_status)

if __name__ == '__main__':
    print("🚀 Starting Enhanced HomeMade GPT Web Interface...")
    print("🌐 Open your browser and go to: http://localhost:5000")
    print("📱 The interface is mobile-friendly!")
    print("🤖 Upload text files to train custom AI models")
    print("💬 Chat with your trained models")
    print("\nPress Ctrl+C to stop the server")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
