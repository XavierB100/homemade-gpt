#!/usr/bin/env python3
"""
Interactive Chat Interface for Enhanced HomeMade GPT
Chat with your trained models!
"""

import os
import argparse
import torch
from contextlib import nullcontext

from ..models.enhanced_gpt import GPT, GPTConfig
from ..training.data_loader import DataProcessor

class ChatBot:
    """Interactive chatbot using trained HomeMade GPT"""
    
    def __init__(self, model_path, device='auto'):
        """Initialize the chatbot with a trained model"""
        
        # Add backward compatibility for old models
        import sys
        from types import ModuleType
        
        # Create fake modules for old imports if they don't exist
        if 'enhanced_gpt' not in sys.modules:
            enhanced_gpt_module = ModuleType('enhanced_gpt')
            enhanced_gpt_module.GPT = GPT
            enhanced_gpt_module.GPTConfig = GPTConfig
            sys.modules['enhanced_gpt'] = enhanced_gpt_module
        
        if 'data_loader' not in sys.modules:
            data_loader_module = ModuleType('data_loader')
            data_loader_module.DataProcessor = DataProcessor
            sys.modules['data_loader'] = data_loader_module
        
        # Device setup
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        
        # Load model with backward compatibility
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Extract model configuration and processor info
        if 'model_args' in checkpoint:
            config = checkpoint['model_args']
        else:
            # Fallback for older checkpoints
            config = GPTConfig(
                vocab_size=checkpoint.get('vocab_size', 50257),
                block_size=checkpoint.get('block_size', 1024)
            )
        
        # Create model
        self.model = GPT(config)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.model.eval()
        
        # Set up processor
        self.processor = DataProcessor()
        if 'processor' in checkpoint:
            # Load processor info from checkpoint
            proc_info = checkpoint['processor']
            self.processor.chars = proc_info['chars']
            self.processor.stoi = proc_info['stoi']
            self.processor.itos = {int(k): v for k, v in proc_info['itos'].items()}
            
            # Recreate encode/decode functions
            self.processor.encode = lambda s: [self.processor.stoi.get(c, 0) for c in s]
            self.processor.decode = lambda l: ''.join([self.processor.itos.get(i, '') for i in l])
        else:
            raise ValueError("No processor information found in checkpoint. Please retrain with the new training script.")
        
        # Mixed precision context
        ptdtype = torch.float16 if device_type == 'cuda' else torch.float32
        self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        
        # Chat history
        self.conversation_history = []
        
        print(f"Model loaded successfully!")
        print(f"Vocabulary size: {len(self.processor.chars)} characters")
        print(f"Model parameters: {self.model.get_num_params()/1e6:.1f}M")
        print(f"Using device: {self.device}")
    
    def generate_response(self, prompt, max_length=200, temperature=0.8, top_k=50, top_p=0.9):
        """Generate a response to the given prompt"""
        
        # Format prompt (add conversation context if available)
        if hasattr(self, 'conversation_format') and self.conversation_format == 'whatsapp':
            # For WhatsApp-trained models, format as conversation
            formatted_prompt = f"User: {prompt}\\nBot:"
        else:
            # For plain text models, use the prompt as-is
            formatted_prompt = prompt
        
        # Encode the prompt
        prompt_ids = self.processor.encode(formatted_prompt)
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Generate response
        with torch.no_grad():
            with self.ctx:
                generated = self.model.generate(
                    prompt_tensor,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
        
        # Decode the generated text
        generated_text = self.processor.decode(generated[0].tolist())
        
        # Extract just the response part (remove the original prompt)
        if formatted_prompt in generated_text:
            response = generated_text[len(formatted_prompt):].strip()
        else:
            response = generated_text.strip()
        
        return response
    
    def chat_loop(self):
        """Main interactive chat loop"""
        print("\\n" + "="*60)
        print("🤖 Enhanced HomeMade GPT Chat Interface")
        print("="*60)
        print("Commands:")
        print("  /help     - Show this help message")
        print("  /reset    - Reset conversation history")
        print("  /temp X   - Set temperature (0.1-2.0)")
        print("  /length X - Set max response length")
        print("  /quit     - Exit chat")
        print("="*60)
        print("Start chatting! (Type /quit to exit)\\n")
        
        # Default generation settings
        temperature = 0.8
        max_length = 200
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    command = user_input.split()[0].lower()
                    
                    if command == '/quit':
                        print("Goodbye! 👋")
                        break
                    
                    elif command == '/help':
                        print("\\nCommands:")
                        print("  /help     - Show this help message")
                        print("  /reset    - Reset conversation history")
                        print("  /temp X   - Set temperature (0.1-2.0)")
                        print("  /length X - Set max response length")
                        print("  /quit     - Exit chat\\n")
                        continue
                    
                    elif command == '/reset':
                        self.conversation_history = []
                        print("Conversation history reset!\\n")
                        continue
                    
                    elif command == '/temp':
                        try:
                            temp = float(user_input.split()[1])
                            if 0.1 <= temp <= 2.0:
                                temperature = temp
                                print(f"Temperature set to {temperature}\\n")
                            else:
                                print("Temperature must be between 0.1 and 2.0\\n")
                        except (IndexError, ValueError):
                            print("Usage: /temp 0.8\\n")
                        continue
                    
                    elif command == '/length':
                        try:
                            length = int(user_input.split()[1])
                            if 10 <= length <= 1000:
                                max_length = length
                                print(f"Max response length set to {max_length}\\n")
                            else:
                                print("Length must be between 10 and 1000\\n")
                        except (IndexError, ValueError):
                            print("Usage: /length 200\\n")
                        continue
                    
                    else:
                        print(f"Unknown command: {command}. Type /help for available commands.\\n")
                        continue
                
                # Generate response
                print("Bot: ", end="", flush=True)
                
                try:
                    response = self.generate_response(
                        user_input, 
                        max_length=max_length,
                        temperature=temperature,
                        top_k=50,
                        top_p=0.9
                    )
                    
                    # Clean up response
                    response = response.strip()
                    if response:
                        print(response)
                        
                        # Add to conversation history
                        self.conversation_history.append({
                            'user': user_input,
                            'bot': response
                        })
                    else:
                        print("[No response generated]")
                
                except Exception as e:
                    print(f"[Error generating response: {e}]")
                
                print()  # Add blank line
                
            except KeyboardInterrupt:
                print("\\n\\nGoodbye! 👋")
                break
            except EOFError:
                print("\\n\\nGoodbye! 👋")
                break

def main():
    parser = argparse.ArgumentParser(description='Chat with Enhanced HomeMade GPT')
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (default: 0.8)')
    parser.add_argument('--max_length', type=int, default=200,
                        help='Maximum response length (default: 200)')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Single prompt to generate from (non-interactive)')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        print("Make sure you've trained a model first using train.py")
        return
    
    # Initialize chatbot
    try:
        chatbot = ChatBot(args.model, args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Single prompt mode
    if args.prompt:
        print(f"Prompt: {args.prompt}")
        response = chatbot.generate_response(
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature
        )
        print(f"Response: {response}")
        return
    
    # Interactive chat mode
    chatbot.chat_loop()

if __name__ == "__main__":
    main()
