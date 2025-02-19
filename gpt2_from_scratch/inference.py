import torch
from model import Model

# Set device priority: CUDA GPU > Apple M1/M2 > CPU
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Set random seed for reproducibility
TOUCH_SEED = 1337
torch.manual_seed(TOUCH_SEED)

# Load and preprocess data (needed for vocabulary)
with open("data/scifi.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Create vocabulary from unique characters
vocab = sorted(list(set(text)))
vocab_size = max_token_value = len(vocab)

# Create character to index and index to character mappings
char2idx = { ch:i for i,ch in enumerate(vocab) }
idx2char = { i:ch for ch, i in char2idx.items() }

# Define encoding and decoding functions
encode = lambda x: [char2idx[c] for c in x]
decode = lambda idxs: ''.join([idx2char[i] for i in idxs])

tekenized_text = torch.tensor(encode(text), dtype=torch.long)

# Load trained model
model = Model(max_token_value=vocab_size).to(device)
model.load_state_dict(torch.load("models/model-scifi.pth"))
model.eval()

# Set up prompt and generate text
prompt = "Jerry was born in a small town."
prompt_ids = encode(prompt)

# Convert prompt to tensor and add batch dimension
x = (torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...])
# Generate continuation
y = model.generate(x, max_new_tokens=500)

# Print generated text
print('-' * 40)
print(decode(y[0].tolist()))
print('-' * 40)
