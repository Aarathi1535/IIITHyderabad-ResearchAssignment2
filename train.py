import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from collections import Counter
import json

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Character-level Dataset for better perplexity on literary text
class TextDataset(Dataset):
    def __init__(self, text_file, seq_length=50, vocab=None, word_to_idx=None, char_level=False):
        with open(text_file, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        self.char_level = char_level
        
        if char_level:
            self.tokens = list(self.text.lower())
        else:
            # Word-level tokenization
            self.tokens = self.text.lower().split()
        
        # Build vocabulary if not provided
        if vocab is None or word_to_idx is None:
            self.build_vocab()
        else:
            self.vocab = vocab
            self.word_to_idx = word_to_idx
        
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.seq_length = seq_length
        self.vocab_size = len(self.vocab)
        
        # Create sequences
        self.sequences = []
        self.targets = []
        for i in range(len(self.tokens) - seq_length):
            seq = self.tokens[i:i + seq_length]
            target = self.tokens[i + seq_length]
            self.sequences.append([self.word_to_idx.get(w, self.word_to_idx['<UNK>']) for w in seq])
            self.targets.append(self.word_to_idx.get(target, self.word_to_idx['<UNK>']))
    
    def build_vocab(self, min_freq=2):
        if self.char_level:
            # For character-level, keep all characters
            self.vocab = ['<PAD>', '<UNK>'] + sorted(list(set(self.tokens)))
        else:
            # For word-level, filter by frequency
            word_counts = Counter(self.tokens)
            self.vocab = ['<PAD>', '<UNK>'] + [word for word, count in word_counts.items() if count >= min_freq]
        
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.targets[idx])

# LSTM Language Model with enhanced architecture
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.5):
        super(LSTMLanguageModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, x, hidden=None):
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = self.dropout(lstm_out[:, -1, :])
        output = self.fc(lstm_out)
        return output, hidden

# Training Function
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output, _ = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % 100 == 0:
            print(f'  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    return total_loss / num_batches

# Validation Function
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

# Calculate Perplexity
def calculate_perplexity(loss):
    return np.exp(loss)

# Main Training Loop
def train_model(config, train_loader, val_loader, device, model_name):
    print(f'\nInitializing {model_name} model...')
    print(f'Config: {config}')
    
    model = LSTMLanguageModel(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    
    train_losses = []
    val_losses = []
    train_perplexities = []
    val_perplexities = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        print(f'\nEpoch {epoch+1}/{config["epochs"]}')
        print('='*60)
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        train_perplexity = calculate_perplexity(train_loss)
        val_perplexity = calculate_perplexity(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_perplexities.append(train_perplexity)
        val_perplexities.append(val_perplexity)
        
        scheduler.step(val_loss)
        
        print(f'Train Loss: {train_loss:.4f}, Train Perplexity: {train_perplexity:.2f}')
        print(f'Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.2f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_perplexity': val_perplexity,
                'config': config
            }, f'models/{model_name}_best.pth')
            print(f'✓ Best model saved (Val Loss: {val_loss:.4f})')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= 5:
            print(f'\nEarly stopping triggered at epoch {epoch+1}')
            break
    
    return model, train_losses, val_losses, train_perplexities, val_perplexities

# Plot Results
def plot_results(train_losses, val_losses, train_perp, val_perp, model_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Training Loss', linewidth=2)
    ax1.plot(val_losses, label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'Training vs Validation Loss - {model_name}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Perplexity plot
    ax2.plot(train_perp, label='Training Perplexity', linewidth=2)
    ax2.plot(val_perp, label='Validation Perplexity', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Perplexity', fontsize=12)
    ax2.set_title(f'Training vs Validation Perplexity - {model_name}', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

# Comparison plot for all models
def plot_comparison(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, (name, data) in enumerate(results.items()):
        ax1.plot(data['val_losses'], label=name, linewidth=2, color=colors[idx])
        ax2.plot(data['val_perp'], label=name, linewidth=2, color=colors[idx])
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation Loss', fontsize=12)
    ax1.set_title('Model Comparison - Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Perplexity', fontsize=12)
    ax2.set_title('Model Comparison - Validation Perplexity', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# Main Execution
if __name__ == '__main__':
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load and prepare dataset
    print('\nLoading Pride and Prejudice dataset...')
    train_dataset = TextDataset('data/train.txt', seq_length=50, char_level=False)
    val_dataset = TextDataset('data/val.txt', seq_length=50, 
                             vocab=train_dataset.vocab, 
                             word_to_idx=train_dataset.word_to_idx,
                             char_level=False)
    
    # Save vocabulary
    with open('models/vocab.json', 'w') as f:
        json.dump(train_dataset.word_to_idx, f)
    
    print(f'\nDataset Statistics:')
    print(f'Vocabulary size: {train_dataset.vocab_size}')
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    
    # Configuration for different scenarios (optimized for Pride and Prejudice)
    configs = {
        'underfit': {
            'vocab_size': train_dataset.vocab_size,
            'embedding_dim': 64,
            'hidden_dim': 128,
            'num_layers': 1,
            'dropout': 0.2,
            'learning_rate': 0.01,
            'epochs': 15,
            'batch_size': 128
        },
        'overfit': {
            'vocab_size': train_dataset.vocab_size,
            'embedding_dim': 512,
            'hidden_dim': 1024,
            'num_layers': 3,
            'dropout': 0.1,
            'learning_rate': 0.0005,
            'epochs': 50,
            'batch_size': 32
        },
        'best_fit': {
            'vocab_size': train_dataset.vocab_size,
            'embedding_dim': 256,
            'hidden_dim': 512,
            'num_layers': 2,
            'dropout': 0.5,
            'learning_rate': 0.001,
            'epochs': 30,
            'batch_size': 64
        }
    }
    
    # Store results for comparison
    all_results = {}
    
    # Train all three models
    for scenario, config in configs.items():
        print(f'\n{"="*70}')
        print(f'Training {scenario.upper()} model')
        print(f'{"="*70}')
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=0)
        
        model, train_losses, val_losses, train_perp, val_perp = train_model(
            config, train_loader, val_loader, device, scenario
        )
        
        plot_results(train_losses, val_losses, train_perp, val_perp, scenario)
        
        all_results[scenario] = {
            'val_losses': val_losses,
            'val_perp': val_perp,
            'final_train_perp': train_perp[-1],
            'final_val_perp': val_perp[-1],
            'best_val_perp': min(val_perp)
        }
        
        print(f'\n{scenario.upper()} Model Results:')
        print(f'Final Training Perplexity: {train_perp[-1]:.2f}')
        print(f'Final Validation Perplexity: {val_perp[-1]:.2f}')
        print(f'Best Validation Perplexity: {min(val_perp):.2f}')
    
    # Create comparison plot
    plot_comparison(all_results)
    
    # Print final summary
    print(f'\n{"="*70}')
    print('FINAL SUMMARY - Pride and Prejudice Language Model')
    print(f'{"="*70}')
    for scenario, results in all_results.items():
        print(f'\n{scenario.upper()}:')
        print(f'  Best Validation Perplexity: {results["best_val_perp"]:.2f}')
        print(f'  Final Validation Perplexity: {results["final_val_perp"]:.2f}')
