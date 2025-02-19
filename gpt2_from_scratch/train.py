import torch
from aim import Run  # Aim for experiment tracking
from model import Model

# Training Hyperparameters
batch_size = 12          # Number of sequences processed in parallel
context_length = 128    # Maximum sequence length
max_iters = 20000      # Total training iterations
learning_rate = 1e-4    # Learning rate for optimizer
eval_interval = 50      # How often to evaluate model
eval_iters = 20        # Number of iterations for evaluation

# Set device priority: CUDA GPU > Apple M1/M2 > CPU
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Set random seed for reproducibility
TOUCH_SEED = 1337
torch.manual_seed(TOUCH_SEED)

# Initialize experiment tracking
run = Run()
run["hparams"] = {
    "learning_rate": learning_rate,
    "max_iters": max_iters,
    "batch_size": batch_size,
    "context_length": context_length,
}

# Load and preprocess data
with open("data/scifi.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Create vocabulary from unique characters
vocab = sorted(list(set(text)))
vocab_size = max_token_value = len(vocab)

# Create character to index and index to character mappings
char2idx = { ch:i for i, ch in enumerate(vocab) }
idx2char = { i:ch for ch, i in char2idx.items()}

# Define encoding and decoding functions
encode = lambda x: [char2idx[c] for c in x]
decode = lambda idxs: "".join([idx2char[i] for i in idxs])

# Convert text to tensor of indices
tokenized_text = torch.tensor(encode(text), dtype=torch.long)

# Split data into train and validation sets
train_size = int(len(tokenized_text) * 0.8)
train_data = tokenized_text[:train_size]
val_data = tokenized_text[train_size:]

# Initialize model
model = Model(max_token_value=vocab_size).to(device)

def get_batch(split: str):
    """Get a random batch of data."""
    # Select appropriate dataset
    data = train_data if split == "train" else val_data
    # Get random starting indices
    idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
    # Get input sequences
    x = torch.stack([data[idx:idx+context_length] for idx in idxs])
    # Get target sequences (shifted by 1)
    y = torch.stack([data[idx+1:idx+context_length+1] for idx in idxs])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    """Estimate loss on train and validation sets."""
    out = {}
    model.eval()
    for split in ["train", "valid"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_batch, y_batch = get_batch(split)
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out    

# Initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
tracked_losses = []

# Training loop
for step_in in range(max_iters):
    # Evaluate model periodically
    if step_in % eval_interval == 0 or step_in == max_iters - 1:
        losses = estimate_loss()
        tracked_losses.append(losses)
        print(f"step {step_in}: training loss {round(losses['train'].item(), 3)}, validation loss {round(losses['valid'].item(),3)}")
        # Log metrics
        run.track(round(losses['train'].item(), 3), name='Training Loss')
        run.track(round(losses['valid'].item(), 3), name='Validation Loss')
    
    # Get batch and compute loss
    x_batch, y_batch = get_batch("train")
    logits, loss = model(x_batch, y_batch)
    
    # Gradient update
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if step_in % eval_interval == 0 or step_in == max_iters - 1:
        print("-" * 100)

# Save model
torch.save(model.state_dict(), "models/model-scifi.pth")
        
    