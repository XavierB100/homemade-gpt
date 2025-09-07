# 🤖 Enhanced HomeMade GPT

> **Train your own AI models on any text data with a beautiful web interface!**

A complete AI training and chat platform that allows users to create personalized AI assistants using their own data - books, WhatsApp conversations, documents, or any text content.

## ✨ Features

### 🌐 **Beautiful Web Interface**
- **Modern Design**: Gradient backgrounds, animations, and professional styling
- **Mobile-Responsive**: Works perfectly on phones, tablets, and desktops
- **Real-Time Updates**: Live training progress with WebSocket connections
- **Drag & Drop**: Simply drag text files to start training

### 🤖 **Smart Training**
- **Any Text Format**: Books, novels, WhatsApp chats, documents
- **Auto-Detection**: Automatically detects and processes different formats
- **Live Progress**: Real-time training progress with loss curves
- **Model Sizes**: From nano (0.8M) to large (150M parameters)
- **Advanced Settings**: Configurable hyperparameters with helpful tooltips

### 💬 **Interactive Chat**
- **ChatGPT-like Interface**: Professional chat interface with your custom models
- **Model Selection**: Easy switching between trained models
- **Creativity Control**: Adjust temperature and response length
- **Chat Export**: Save conversations as text files
- **Typing Indicators**: Visual feedback while AI is thinking

### 🛠️ **Model Management**
- **Beautiful Dashboard**: Visual cards showing all your models
- **Statistics**: Total models, storage usage, working vs error models
- **Model Info**: Detailed information about each model
- **Safe Deletion**: Model deletion with confirmation prompts

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/XavierB100/homemade-gpt.git
   cd homemade-gpt
   ```

2. **Install dependencies**:
   ```bash
   pip install torch flask flask-socketio
   ```

3. **Start the web application**:
   ```bash
   python web_app.py
   ```

4. **Open your browser**:
   ```
   http://localhost:5000
   ```

### First Steps

1. **Train Your First Model**:
   - Go to the "Train" page
   - Drag & drop any .txt file (book, WhatsApp export, etc.)
   - Choose model size and settings
   - Watch real-time training progress!

2. **Chat with Your AI**:
   - Go to the "Chat" page
   - Select your trained model
   - Start chatting!

## 📚 What You Can Train On

### 📖 **Books & Novels**
- Upload any book in .txt format
- Train AI to write in that author's style
- Generate new chapters or continue stories
- Chat about themes and characters

**Example**: Train on Shakespeare's works, then generate new sonnets or discuss literary themes.

### 💬 **WhatsApp Conversations**
- Export WhatsApp chat (without media)
- Train AI on your conversation patterns
- Get responses in your group's style
- Continue conversations naturally

**How to export WhatsApp**:
- **Android**: Chat → ⋮ → More → Export chat → Without media
- **iPhone**: Chat → Contact name → Export Chat → Without Media

### 📄 **Any Text Content**
- Documentation, articles, code
- Personal writing, journals, notes
- Create domain-specific AI assistants
- Generate content in your specific style

## 🎯 Model Sizes & Performance

| Size   | Parameters | Best For                    | Training Time | Memory Usage |
|--------|------------|----------------------------|---------------|--------------|
| Nano   | ~0.8M      | Quick experiments          | Fast          | Low          |
| Micro  | ~1.5M      | Small datasets             | Fast          | Low          |
| Tiny   | ~10M       | Most use cases (recommended)| Moderate      | Moderate     |
| Small  | ~25M       | Better quality             | Slower        | Higher       |
| Medium | ~70M       | High quality               | Slow          | High         |
| Large  | ~150M      | Best quality (if you have time)| Very Slow     | Very High    |

## 🔧 Architecture & Improvements

This enhanced version includes modern transformer improvements over the original nanogpt:

### **Technical Enhancements**
- **Pre-normalization**: LayerNorm before attention (more stable training)
- **Better initialization**: Proper weight scaling for faster convergence
- **Flash Attention**: Memory-efficient attention computation when available
- **GELU activation**: More natural than ReLU
- **Advanced optimizer**: AdamW with proper weight decay
- **Learning rate scheduling**: Cosine annealing with warmup

### **Web Application Stack**
- **Backend**: Flask + SocketIO for real-time updates
- **Frontend**: Bootstrap 5 with custom CSS for modern design
- **Real-time**: WebSocket integration for live training progress
- **File handling**: Secure upload with validation
- **Data processing**: Automatic format detection and preprocessing

## 📊 Training Tips

### **For Books/Novels**
- **Minimum**: 1MB of text (short book)
- **Recommended**: 5MB+ (several books or long novel)
- **Model**: `small` or `medium` for best results
- **Iterations**: 5000-10000

### **For WhatsApp Conversations**
- **Minimum**: 500 messages
- **Recommended**: 2000+ messages
- **Model**: `tiny` or `small`
- **Iterations**: 2000-5000

### **General Guidelines**
- More data = better results
- Larger models need more training time
- Start with `tiny` model for testing
- Use GPU for faster training if available

## 🎨 Screenshots

The web interface features:
- **Beautiful Homepage**: Feature showcase with model overview
- **Training Page**: Drag & drop upload with real-time progress
- **Chat Interface**: Modern chat bubbles with typing indicators
- **Model Dashboard**: Visual cards with statistics and management

## 🤝 Contributing

This is an educational project! Feel free to:
- Add new data formats
- Improve the web interface
- Optimize training speed
- Add new model architectures

## 📝 License

MIT License - same as the original nanogpt project.

## 🙏 Acknowledgments

Built upon the excellent [nanogpt](https://github.com/karpathy/nanoGPT) project by Andrej Karpathy. This enhanced version adds:
- Beautiful web interface
- Support for multiple data formats
- Modern transformer improvements
- Real-time training progress
- Interactive chat functionality

## 🔗 Links

- **Original nanogpt**: https://github.com/karpathy/nanoGPT
- **Neural Networks: Zero To Hero**: https://karpathy.ai/zero-to-hero.html

---

**🚀 Ready to build your own AI? Clone this repo and start training!**

*Made with ❤️ for the AI community*
