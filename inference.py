import torch
import json
from train import LSTMLanguageModel, set_seed
import random

set_seed(42)

def load_model(model_path, vocab_path, device):
    """Load trained model and vocabulary"""
    with open(vocab_path, 'r') as f:
        word_to_idx = json.load(f)
    idx_to_word = {int(idx): word for word, idx in word_to_idx.items()}
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model = LSTMLanguageModel(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=0.0
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, word_to_idx, idx_to_word, checkpoint

def generate_text(model, seed_text, word_to_idx, idx_to_word, length=100, temperature=1.0, device='cpu'):
    """Generate text using the trained model"""
    model.eval()
    tokens = seed_text.lower().split()
    
    with torch.no_grad():
        for _ in range(length):
            # Prepare input (last 50 tokens or less)
            input_seq = [word_to_idx.get(w, word_to_idx['<UNK>']) for w in tokens[-50:]]
            
            # Pad if necessary
            while len(input_seq) < 50:
                input_seq = [word_to_idx['<PAD>']] + input_seq
            
            input_tensor = torch.tensor([input_seq]).to(device)
            
            # Get prediction
            output, _ = model(input_tensor)
            
            # Apply temperature
            output = output / temperature
            probs = torch.softmax(output, dim=1)
            
            # Sample next word
            next_idx = torch.multinomial(probs, 1).item()
            next_word = idx_to_word[next_idx]
            
            # Skip special tokens
            if next_word in ['<PAD>', '<UNK>']:
                continue
            
            tokens.append(next_word)
    
    return ' '.join(tokens)

def complete_sentence(model, seed_text, word_to_idx, idx_to_word, max_length=50, device='cpu'):
    """Complete a sentence naturally"""
    model.eval()
    tokens = seed_text.lower().split()
    
    with torch.no_grad():
        for _ in range(max_length):
            input_seq = [word_to_idx.get(w, word_to_idx['<UNK>']) for w in tokens[-50:]]
            
            while len(input_seq) < 50:
                input_seq = [word_to_idx['<PAD>']] + input_seq
            
            input_tensor = torch.tensor([input_seq]).to(device)
            output, _ = model(input_tensor)
            
            # Use greedy decoding for sentence completion
            next_idx = torch.argmax(output, dim=1).item()
            next_word = idx_to_word[next_idx]
            
            if next_word in ['<PAD>', '<UNK>']:
                continue
            
            tokens.append(next_word)
            
            # Stop at sentence end
            if next_word.endswith('.') or next_word.endswith('!') or next_word.endswith('?'):
                break
    
    return ' '.join(tokens)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load best model
    model, word_to_idx, idx_to_word, checkpoint = load_model(
        'models/best_fit_best.pth',
        'models/vocab.json',
        device
    )
    
    print(f'Loaded model from epoch {checkpoint["epoch"]}')
    print(f'Validation Perplexity: {checkpoint["val_perplexity"]:.2f}')
    print('='*70)
    
    # Example 1: Generate text with different temperatures
    seed_texts = [
        "it is a truth universally acknowledged",
        "mr darcy was",
        "elizabeth bennet could not"
    ]
    
    for seed in seed_texts:
        print(f'\nSeed: "{seed}"')
        print('-'*70)
        
        # Conservative generation
        conservative = generate_text(model, seed, word_to_idx, idx_to_word, length=30, temperature=0.7, device=device)
        print(f'\nConservative (temp=0.7):\n{conservative}')
        
        # Creative generation
        creative = generate_text(model, seed, word_to_idx, idx_to_word, length=30, temperature=1.2, device=device)
        print(f'\nCreative (temp=1.2):\n{creative}')
        
        print('='*70)
    
    # Example 2: Sentence completion
    print('\n\nSENTENCE COMPLETION:')
    print('='*70)
    incomplete_sentences = [
        "the family of dashwood had long been",
        "she was determined to",
        "in spite of all"
    ]
    
    for incomplete in incomplete_sentences:
        completed = complete_sentence(model, incomplete, word_to_idx, idx_to_word, device=device)
        print(f'\nInput: "{incomplete}"')
        print(f'Completed: {completed}')
        print('-'*70)
