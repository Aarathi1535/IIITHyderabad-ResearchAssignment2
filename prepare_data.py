import os
import re

def clean_text(text):
    """Clean Pride and Prejudice text from Project Gutenberg"""
    # Find the start and end markers
    start_marker = "***START OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE***"
    end_marker = "***END OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE***"
    
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    
    if start_idx != -1:
        text = text[start_idx + len(start_marker):]
    if end_idx != -1:
        text = text[:end_idx]
    
    # Remove chapter headers and illustration descriptions
    text = re.sub(r'\[Illustration:.*?\]', '', text, flags=re.DOTALL)
    text = re.sub(r'CHAPTER [IVXLCDM]+\.', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Remove underscores used for italics
    text = text.replace('_', '')
    
    # Remove special characters for superscripts
    text = re.sub(r'\^{.*?}', '', text)
    
    return text.strip()

def split_dataset(input_file, train_ratio=0.8, val_ratio=0.1):
    """Split Pride and Prejudice into train, validation, and test sets"""
    os.makedirs('data', exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Clean the text
    text = clean_text(text)
    
    # Split into sentences for better segmentation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Calculate split sizes
    total = len(sentences)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_text = ' '.join(sentences[:train_size])
    val_text = ' '.join(sentences[train_size:train_size + val_size])
    test_text = ' '.join(sentences[train_size + val_size:])
    
    with open('data/train.txt', 'w', encoding='utf-8') as f:
        f.write(train_text)
    
    with open('data/val.txt', 'w', encoding='utf-8') as f:
        f.write(val_text)
    
    with open('data/test.txt', 'w', encoding='utf-8') as f:
        f.write(test_text)
    
    print(f'Dataset processed: Pride and Prejudice by Jane Austen')
    print(f'Total sentences: {total}')
    print(f'Train sentences: {train_size}')
    print(f'Validation sentences: {val_size}')
    print(f'Test sentences: {total - train_size - val_size}')
    print(f'\nTrain words: {len(train_text.split())}')
    print(f'Validation words: {len(val_text.split())}')
    print(f'Test words: {len(test_text.split())}')

if __name__ == '__main__':
    split_dataset('Pride_and_Prejudice-Jane_Austen.txt')
