import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import re
from transformer import SimpleTransformer

class TextDataset(Dataset):
    """Simple text dataset for demonstration"""
    def __init__(self, texts, labels, vocab, max_length=128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Simple tokenization and encoding
        tokens = self.tokenize(text)
        encoded = self.encode_tokens(tokens)

        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def tokenize(self, text):
        # Simple tokenization: lowercase and split on whitespace/punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return tokens[:self.max_length]

    def encode_tokens(self, tokens):
        # Encode tokens to indices, pad to max_length
        encoded = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]

        # Pad or truncate to max_length
        if len(encoded) < self.max_length:
            encoded.extend([self.vocab['<PAD>']] * (self.max_length - len(encoded)))
        else:
            encoded = encoded[:self.max_length]

        return encoded

def build_vocab(texts, min_freq=2, max_vocab_size=10000):
    """Build vocabulary from texts"""
    # Tokenize all texts and count words
    word_counts = Counter()
    for text in texts:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        word_counts.update(tokens)

    # Create vocabulary
    vocab = {'<PAD>': 0, '<UNK>': 1}

    # Add most frequent words
    most_common = word_counts.most_common(max_vocab_size - 2)
    for word, count in most_common:
        if count >= min_freq:
            vocab[word] = len(vocab)

    return vocab

def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        if batch_idx % 50 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}')

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} - Average Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%')

    return avg_loss, accuracy

def test(model, device, test_loader, criterion):
    """Test the model"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    print(f'Test - Average Loss: {avg_loss:.6f}, '
          f'Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')

    return avg_loss, accuracy

def create_sample_data(num_samples=1000, num_classes=5):
    """Create sample text data for demonstration"""
    # Create simple synthetic data
    texts = []
    labels = []

    # Simple patterns for different classes
    patterns = [
        ["computer", "software", "technology", "programming", "code"],
        ["sports", "football", "basketball", "game", "team"],
        ["science", "research", "experiment", "theory", "study"],
        ["music", "song", "artist", "album", "concert"],
        ["food", "restaurant", "cooking", "recipe", "delicious"]
    ]

    for i in range(num_samples):
        class_idx = i % num_classes
        # Create a simple sentence using words from the pattern
        words = np.random.choice(patterns[class_idx], size=np.random.randint(5, 15))
        text = " ".join(words)
        texts.append(text)
        labels.append(class_idx)

    return texts, labels

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Training parameters
    batch_size = 32
    epochs = 10
    learning_rate = 0.001
    max_length = 64

    # Create sample data (in practice, you would load real text data)
    print("Creating sample data...")
    texts, labels = create_sample_data(num_samples=2000, num_classes=5)

    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Build vocabulary
    print("Building vocabulary...")
    vocab = build_vocab(train_texts, min_freq=1, max_vocab_size=5000)
    vocab_size = len(vocab)
    num_classes = len(set(labels))

    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of classes: {num_classes}")

    # Create datasets and dataloaders
    train_dataset = TextDataset(train_texts, train_labels, vocab, max_length)
    test_dataset = TextDataset(test_texts, test_labels, vocab, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        num_classes=num_classes,
        max_length=max_length
    ).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\nModel has {sum(p.numel() for p in model.parameters()):,} parameters")
    print("Starting training...\n")

    # Training loop
    best_test_acc = 0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, criterion, epoch)
        test_loss, test_acc = test(model, device, test_loader, criterion)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'transformer_best.pth')
            print(f"New best model saved with test accuracy: {best_test_acc:.2f}%")

    print(f"\nTraining completed. Best test accuracy: {best_test_acc:.2f}%")

if __name__ == '__main__':
    main()
