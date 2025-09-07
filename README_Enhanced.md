# Enhanced HomeMade GPT 🤖

An improved version of the original nanogpt project with support for custom training data including **plain text (books, novels)** and **WhatsApp conversations**!

## ✨ What's New

- **📚 Flexible Data Support**: Train on books, novels, WhatsApp chats, or any text
- **🏗️ Modern Architecture**: Pre-normalization, better weight initialization, Flash Attention
- **⚙️ Configurable Models**: Multiple size presets (nano to large)
- **🚀 Advanced Training**: Learning rate scheduling, gradient clipping, checkpointing
- **💬 Interactive Chat**: Built-in chat interface with conversation memory
- **🔧 Easy to Use**: Simple command-line interface

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install torch
```

### 2. Create Sample Data (Optional)
```bash
# Generate sample WhatsApp conversation
python train.py --create_sample
```

### 3. Train Your Model
```bash
# Train on Shakespeare (existing data)
python train.py --data input.txt --model_size tiny --max_iters 2000

# Train on your own book/novel
python train.py --data your_book.txt --model_size small --max_iters 5000

# Train on WhatsApp conversation
python train.py --data your_whatsapp_export.txt --model_size tiny --max_iters 3000
```

### 4. Chat with Your Model
```bash
python chat.py --model models/ckpt.pt
```

## 📖 Detailed Usage

### Training Options

```bash
python train.py [OPTIONS]

Required:
  --data PATH              Path to your training data file

Model Configuration:
  --model_size SIZE        Model size: nano, micro, tiny, small, medium, large
  --block_size INT         Context length (default: 256)
  --dropout FLOAT          Dropout rate (default: 0.1)

Training Settings:
  --batch_size INT         Batch size (default: 64)
  --max_iters INT          Training iterations (default: 5000)
  --learning_rate FLOAT    Learning rate (default: 3e-4)
  --device DEVICE          Device: auto, cpu, cuda

Other:
  --data_type TYPE         Data format: auto, plain_text, whatsapp
  --output_dir PATH        Where to save models (default: ./models)
```

### Model Sizes

| Size   | Parameters | Layers | Embedding | Good For                    |
|--------|------------|--------|-----------|----------------------------|
| nano   | ~0.8M      | 3      | 144       | Quick experiments          |
| micro  | ~1.5M      | 4      | 192       | Small datasets             |
| tiny   | ~10M       | 6      | 384       | Books, conversations       |
| small  | ~25M       | 8      | 512       | Larger texts               |
| medium | ~70M       | 12     | 768       | Complex writing styles     |
| large  | ~150M      | 16     | 1024      | Maximum quality (slow)     |

### Chat Interface

```bash
python chat.py --model models/ckpt.pt

Interactive Commands:
  /help          Show help
  /temp 0.8      Set creativity (0.1-2.0)
  /length 200    Set response length
  /reset         Clear conversation
  /quit          Exit

# Single prompt mode
python chat.py --model models/ckpt.pt --prompt "Once upon a time"
```

## 📱 WhatsApp Training Data

### Exporting WhatsApp Chats

1. **Android**: Open chat → ⋮ → More → Export chat → Without media
2. **iPhone**: Open chat → Contact name → Export Chat → Without Media

### Supported Formats

The system auto-detects these WhatsApp timestamp formats:
- `12/31/23, 10:30 AM - Alice: Hello`
- `[12/31/23, 10:30:45] Bob: How are you?`
- `31-12-2023 10:30 - Charlie: Great!`

## 📚 Training Tips

### For Books/Novels
- **Minimum**: 1MB of text (short book)
- **Recommended**: 5MB+ (several books or long novel)
- **Model**: `small` or `medium` for best results
- **Iterations**: 5000-10000

### For WhatsApp Conversations  
- **Minimum**: 500 messages
- **Recommended**: 2000+ messages
- **Model**: `tiny` or `small`
- **Iterations**: 2000-5000

### General Guidelines
- More data = better results
- Larger models need more training time
- Start with `tiny` model for testing
- Use GPU for faster training

## 🎯 Example Training Sessions

### Train on a Novel
```bash
# Download any public domain book
python train.py --data pride_and_prejudice.txt --model_size small --max_iters 5000
```

### Train on WhatsApp Chat
```bash
# Use your exported chat
python train.py --data whatsapp_chat.txt --data_type whatsapp --model_size tiny --max_iters 3000
```

### Train on Code/Documentation
```bash
# Train on Python code or docs
python train.py --data python_tutorials.txt --model_size medium --max_iters 7000
```

## 🔧 Advanced Features

### Custom Training Loop
The training script includes:
- **Learning rate scheduling** with warmup and cosine decay
- **Gradient clipping** for training stability
- **Mixed precision training** for speed
- **Automatic checkpointing** saves best models
- **Validation tracking** monitors overfitting

### Generation Options
- **Temperature**: Controls randomness (0.1=focused, 2.0=creative)
- **Top-k**: Only consider top k most likely tokens
- **Top-p**: Nucleus sampling for natural text

## 🐛 Troubleshooting

### Common Issues

**"No module named 'torch'"**
```bash
pip install torch
```

**"CUDA out of memory"**
- Use smaller `--batch_size` (try 32, 16, or 8)
- Use smaller `--model_size`
- Use `--device cpu`

**Poor generation quality**
- Train for more iterations
- Use more training data
- Try a larger model size
- Adjust temperature in chat

**Training too slow**
- Use `--device cuda` if you have GPU
- Reduce `--block_size` to 128
- Use smaller model size for testing

## 💡 What Your Model Can Do

After training, your HomeMade GPT can:

### 📖 For Book-Trained Models
- Continue stories in the author's style
- Generate new chapters or scenes
- Answer questions about the book's world
- Create character dialogues

### 💬 For WhatsApp-Trained Models  
- Chat in your conversation style
- Respond like people in your group
- Continue conversations naturally
- Remember conversation patterns

### 📝 General Capabilities
- Complete text prompts
- Generate creative content
- Maintain consistent style
- Follow learned patterns

## 🏗️ Architecture Improvements

This enhanced version includes:

- **Pre-normalization**: LayerNorm before attention (more stable)
- **Better initialization**: Proper weight scaling for faster convergence  
- **Flash Attention**: Memory-efficient attention computation
- **GELU activation**: More natural than ReLU
- **Configurable architecture**: Easy model size adjustments
- **Modern optimizer**: AdamW with weight decay

## 🤝 Contributing

This is an educational project! Feel free to:
- Add new data formats
- Improve the chat interface
- Optimize training speed
- Add new model architectures

## 📜 License

MIT License - same as the original nanogpt project.

---

**Happy training! 🎉 Create your own personal AI assistant!**
