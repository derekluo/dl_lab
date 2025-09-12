import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import math


class ImageEncoder(nn.Module):
    """Vision Transformer (ViT) based image encoder"""

    def __init__(self, image_size=224, patch_size=16, embed_dim=512, num_heads=8, num_layers=12):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Class token and position embeddings
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

        # Transformer encoder layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_layers
        )

        # Layer normalization
        self.ln_final = nn.LayerNorm(embed_dim)

        # Projection head
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Add class token
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_tokens, x], dim=1)

        # Add position embeddings
        x = x + self.pos_embed

        # Apply transformer
        x = self.transformer(x)

        # Use class token for final representation
        x = self.ln_final(x[:, 0])

        # Apply projection
        x = self.projection(x)

        # L2 normalize
        x = F.normalize(x, dim=-1)

        return x


class TextEncoder(nn.Module):
    """Transformer based text encoder"""

    def __init__(self, vocab_size=49408, embed_dim=512, num_heads=8, num_layers=12, max_length=77):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.randn(max_length, embed_dim))

        # Transformer encoder layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_layers
        )

        # Layer normalization
        self.ln_final = nn.LayerNorm(embed_dim)

        # Projection head
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # Token embedding
        x = self.token_embed(x)

        # Add position embeddings
        seq_len = x.shape[1]
        x = x + self.pos_embed[:seq_len]

        # Apply transformer
        x = self.transformer(x)

        # Use the last token for final representation (EOS token)
        x = self.ln_final(x[:, -1])

        # Apply projection
        x = self.projection(x)

        # L2 normalize
        x = F.normalize(x, dim=-1)

        return x


class CLIP(nn.Module):
    """CLIP model combining image and text encoders"""

    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 embed_dim=512,
                 num_heads=8,
                 num_layers=12,
                 vocab_size=49408,
                 max_text_length=77):
        super().__init__()

        # Image encoder
        self.image_encoder = ImageEncoder(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )

        # Text encoder
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_length=max_text_length
        )

        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, images, texts):
        # Get image and text features
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(texts)

        # Compute similarity matrix
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

    def encode_image(self, images):
        """Encode images separately"""
        return self.image_encoder(images)

    def encode_text(self, texts):
        """Encode texts separately"""
        return self.text_encoder(texts)


class CLIPLoss(nn.Module):
    """Contrastive loss for CLIP training"""

    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits_per_image, logits_per_text):
        # Create target labels (diagonal matrix)
        batch_size = logits_per_image.shape[0]
        targets = torch.arange(batch_size, device=logits_per_image.device)

        # Compute symmetric loss
        loss_i2t = self.cross_entropy(logits_per_image, targets)
        loss_t2i = self.cross_entropy(logits_per_text, targets)

        return (loss_i2t + loss_t2i) / 2


def create_clip_model(model_size='base'):
    """Create CLIP model with different sizes"""

    configs = {
        'tiny': {
            'embed_dim': 256,
            'num_heads': 4,
            'num_layers': 6,
        },
        'small': {
            'embed_dim': 384,
            'num_heads': 6,
            'num_layers': 8,
        },
        'base': {
            'embed_dim': 512,
            'num_heads': 8,
            'num_layers': 12,
        },
        'large': {
            'embed_dim': 768,
            'num_heads': 12,
            'num_layers': 24,
        }
    }

    config = configs.get(model_size, configs['base'])
    return CLIP(**config)


# Text tokenization utilities
class SimpleTokenizer:
    """Simple tokenizer for CLIP text encoding"""

    def __init__(self, max_length=77):
        self.max_length = max_length
        # Use a simple character-level tokenizer for demonstration
        # In practice, you'd use a proper BPE tokenizer
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0

        # Build vocabulary from common characters
        chars = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
        for i, char in enumerate(chars):
            self.char_to_idx[char] = i
            self.idx_to_char[i] = char

        # Special tokens
        self.char_to_idx['<SOS>'] = len(chars)
        self.char_to_idx['<EOS>'] = len(chars) + 1
        self.char_to_idx['<PAD>'] = len(chars) + 2
        self.char_to_idx['<UNK>'] = len(chars) + 3

        self.idx_to_char[len(chars)] = '<SOS>'
        self.idx_to_char[len(chars) + 1] = '<EOS>'
        self.idx_to_char[len(chars) + 2] = '<PAD>'
        self.idx_to_char[len(chars) + 3] = '<UNK>'

        self.vocab_size = len(chars) + 4

    def encode(self, text):
        """Encode text to token IDs"""
        if isinstance(text, str):
            text = [text]

        encoded = []
        for t in text:
            # Convert to lowercase and limit length
            t = t.lower()[:self.max_length - 2]  # Reserve space for SOS and EOS

            # Convert characters to indices
            tokens = [self.char_to_idx['<SOS>']]
            for char in t:
                if char in self.char_to_idx:
                    tokens.append(self.char_to_idx[char])
                else:
                    tokens.append(self.char_to_idx['<UNK>'])
            tokens.append(self.char_to_idx['<EOS>'])

            # Pad to max length
            while len(tokens) < self.max_length:
                tokens.append(self.char_to_idx['<PAD>'])

            encoded.append(tokens)

        return torch.tensor(encoded, dtype=torch.long)

    def decode(self, tokens):
        """Decode token IDs to text"""
        if len(tokens.shape) == 1:
            tokens = tokens.unsqueeze(0)

        decoded = []
        for token_seq in tokens:
            text = ""
            for token in token_seq:
                char = self.idx_to_char.get(token.item(), '<UNK>')
                if char in ['<SOS>', '<EOS>', '<PAD>']:
                    if char == '<EOS>':
                        break
                    continue
                text += char
            decoded.append(text)

        return decoded if len(decoded) > 1 else decoded[0]
