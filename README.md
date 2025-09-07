# HomeMade GPT Web Application

🧠 **An interactive web application for training and chatting with custom GPT-style language models**

Build your own AI models from scratch with a beautiful, modern web interface. Upload text data, train neural networks, and chat with your custom AI models - all through your browser!

![Modern AI Interface](https://img.shields.io/badge/Interface-Modern%20Dark%20Theme-blueviolet)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-green)

## ✨ Features

### 🚀 **Complete Web Experience**
- **Beautiful Modern UI**: Dark-themed interface with purple/cyan accents
- **Real-time Training**: Watch your model train with live progress updates
- **Interactive Chat**: Chat with your trained models instantly
- **Model Management**: View, analyze, and manage all your trained models
- **Drag & Drop Upload**: Easy file uploads with visual feedback

### 🤖 **AI Capabilities**
- **Custom Model Training**: Train on your own text data (books, chats, documents)
- **Multiple Text Formats**: Support for plain text, chat logs, and more
- **Character-level GPT**: Educational transformer implementation
- **Real-time Inference**: Fast text generation and chat responses
- **Model Persistence**: Save and load trained models automatically

### 📊 **Professional Features**
- **Progress Monitoring**: Real-time loss tracking and iteration counters
- **Model Analytics**: View model size, parameters, vocabulary stats
- **Error Handling**: Graceful error displays with detailed information
- **Responsive Design**: Works perfectly on desktop and mobile

## 🛠️ **Quick Start**

### Prerequisites
```bash
# Ensure you have Python 3.8+ installed
python --version
```

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
python web_app.py
```

### 3. Open Your Browser
Navigate to: **http://127.0.0.1:5000**

🎉 **That's it! Your AI training lab is ready!**

## 📖 **How to Use**

### 🎯 **Training Your First Model**
1. **Go to Training Page** → Upload a text file (.txt)
2. **Configure Settings** → Set model name and training parameters  
3. **Start Training** → Watch real-time progress with live updates
4. **Monitor Progress** → View loss curves and training statistics

### 💬 **Chatting with Models**
1. **Go to Chat Page** → Select your trained model
2. **Start Conversation** → Type messages and get AI responses
3. **Adjust Settings** → Control response length and creativity

### 📋 **Managing Models**
1. **Go to Models Page** → View all your trained models
2. **Check Details** → Click "Info" to see model statistics
3. **Clean Up** → Delete models you no longer need

## 📁 **Sample Data Included**

Get started immediately with included sample datasets:
- **`input.txt`** - Complete Shakespeare works (~1MB of text)
- **`more.txt`** - Additional literary texts for training
- **`sample_whatsapp.txt`** - WhatsApp chat format example

## 🏗️ **Project Architecture**

```
HomeMadeGPT/
├── 🌐 web_app.py           # Main Flask application
├── 🧠 enhanced_gpt.py      # Neural network model
├── 🚀 train.py             # Training engine
├── 💬 chat.py              # Chat interface
├── 📊 data_loader.py       # Data processing
├── 📄 templates/           # Web interface templates
├── 🤖 models/              # Trained model storage
├── 📁 uploads/             # User uploaded files
├── 📋 logs/                # Training logs
└── 📦 requirements.txt     # Dependencies
```

## 🎨 **Interface Features**

### Modern Dark Theme
- **Professional styling** with deep purple backgrounds
- **Cyan accent colors** for interactive elements
- **Clean typography** with monospace fonts for technical data
- **Responsive cards** and modern button designs

### Real-time Training
- **Live progress bars** showing training completion
- **Real-time loss graphs** updating every iteration
- **WebSocket connectivity** for instant updates
- **Training logs** with color-coded messages

## 🔧 **Technical Details**

### Model Architecture
- **Transformer-based** character-level language model
- **Multi-head attention** with configurable layers
- **Positional encoding** for sequence understanding
- **Layer normalization** and dropout for stability

### Training Features
- **Gradient accumulation** for effective large batch training
- **Learning rate scheduling** with warmup periods
- **Automatic checkpointing** every N iterations
- **Memory optimization** for training on consumer hardware

### Web Technology
- **Flask** web framework with **SocketIO** for real-time updates
- **Bootstrap 5** with custom dark theme styling
- **WebSocket** communication for live training updates
- **Responsive design** that works on all devices

## 🎓 **Educational Purpose**

Perfect for learning:
- ✅ **How transformers work** - See attention mechanisms in action
- ✅ **Neural network training** - Watch loss curves and understand convergence
- ✅ **Web-based ML** - Learn to build interfaces for AI models
- ✅ **Real-time systems** - Implement live updates and monitoring

## 🚀 **What You Can Build**

- **📚 Book-style AI**: Train on novels to generate similar prose
- **💬 Chat personalities**: Create AI that mimics conversation styles  
- **📝 Writing assistants**: Generate text in specific formats or styles
- **🎭 Character models**: Train on dialogue to create virtual characters

## 🛠️ **Customization**

Easily modify:
- **Model parameters**: Change layers, attention heads, embedding size
- **Training settings**: Adjust batch size, learning rate, iterations
- **UI theme**: Customize colors and styling in templates
- **Text processing**: Add support for new data formats

## 📈 **Performance**

- **Fast training**: Optimized for both CPU and GPU
- **Memory efficient**: Trains models on modest hardware
- **Real-time inference**: Quick response times for chat
- **Scalable architecture**: Easy to extend and modify

## 🔧 **File Structure After Cleanup**

Essential files for web application:
- ✅ `web_app.py` - Main Flask app
- ✅ `enhanced_gpt.py` - AI model 
- ✅ `train.py` - Training logic
- ✅ `chat.py` - Chat functionality
- ✅ `data_loader.py` - Data processing
- ✅ `templates/` - Web templates
- ✅ `requirements.txt` - Dependencies
- ✅ Sample data files for immediate experimentation

---

**🎯 Start building your own AI models today!** Upload some text, train a model, and chat with your creation - it's that simple!

*This project is designed for education and experimentation. Perfect for learning how modern language models work under the hood.*
