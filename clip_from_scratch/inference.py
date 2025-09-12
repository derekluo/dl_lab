import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import json

from model import CLIP, SimpleTokenizer, create_clip_model


class CLIPInference:
    """CLIP model for inference tasks"""

    def __init__(self, model_path, device='auto'):
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model_size = checkpoint.get('model_size', 'base')

        # Create model
        self.model = create_clip_model(model_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Create tokenizer and transforms
        self.tokenizer = SimpleTokenizer()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        print(f"Loaded CLIP model ({model_size}) on {self.device}")
        print(f"Model trained for {checkpoint['epoch']} epochs")
        print(f"Best validation loss: {checkpoint['val_loss']:.4f}")

    def encode_image(self, image_path):
        """Encode a single image"""
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path  # PIL Image

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)

        return image_features

    def encode_text(self, text):
        """Encode text or list of texts"""
        if isinstance(text, str):
            text = [text]

        text_tokens = self.tokenizer.encode(text).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)

        return text_features

    def compute_similarity(self, image_features, text_features):
        """Compute cosine similarity between image and text features"""
        # Features are already normalized
        similarity = image_features @ text_features.t()
        return similarity

    def image_text_similarity(self, image_path, texts):
        """Compute similarity between an image and multiple texts"""
        image_features = self.encode_image(image_path)
        text_features = self.encode_text(texts)

        similarity = self.compute_similarity(image_features, text_features)
        return similarity.squeeze(0).cpu().numpy()

    def text_image_retrieval(self, text, image_paths):
        """Rank images by similarity to text query"""
        text_features = self.encode_text([text])

        similarities = []
        for image_path in image_paths:
            image_features = self.encode_image(image_path)
            similarity = self.compute_similarity(image_features, text_features)
            similarities.append(similarity.item())

        # Sort by similarity
        ranked_indices = np.argsort(similarities)[::-1]
        ranked_similarities = [similarities[i] for i in ranked_indices]
        ranked_images = [image_paths[i] for i in ranked_indices]

        return ranked_images, ranked_similarities

    def image_text_retrieval(self, image_path, texts):
        """Rank texts by similarity to image query"""
        image_features = self.encode_image(image_path)
        text_features = self.encode_text(texts)

        similarities = self.compute_similarity(image_features, text_features).cpu().numpy()

        # Sort by similarity
        ranked_indices = np.argsort(similarities)[::-1]
        ranked_similarities = similarities[ranked_indices]
        ranked_texts = [texts[i] for i in ranked_indices]

        return ranked_texts, ranked_similarities

    def zero_shot_classification(self, image_path, class_names, template="a photo of a {}"):
        """Perform zero-shot image classification"""
        # Create text prompts
        texts = [template.format(class_name) for class_name in class_names]

        # Compute similarities
        similarities = self.image_text_similarity(image_path, texts)

        # Apply softmax to get probabilities
        probabilities = F.softmax(torch.tensor(similarities), dim=0).numpy()

        # Sort by probability
        sorted_indices = np.argsort(probabilities)[::-1]

        results = []
        for i in sorted_indices:
            results.append({
                'class': class_names[i],
                'probability': probabilities[i],
                'similarity': similarities[i]
            })

        return results


def visualize_results(image_path, texts, similarities, save_path=None):
    """Visualize image-text similarity results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Display image
    image = Image.open(image_path)
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Query Image')

    # Display similarity scores
    colors = plt.cm.viridis(np.linspace(0, 1, len(texts)))
    bars = ax2.barh(range(len(texts)), similarities, color=colors)
    ax2.set_yticks(range(len(texts)))
    ax2.set_yticklabels(texts)
    ax2.set_xlabel('Similarity Score')
    ax2.set_title('Text Similarity Scores')
    ax2.grid(axis='x', alpha=0.3)

    # Add value labels on bars
    for i, (bar, sim) in enumerate(zip(bars, similarities)):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{sim:.3f}', va='center', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Results saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='CLIP Inference')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained CLIP model')
    parser.add_argument('--mode', type=str, default='similarity',
                      choices=['similarity', 'classification', 'retrieval'],
                      help='Inference mode')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--text', type=str, help='Input text query')
    parser.add_argument('--texts', type=str, nargs='+',
                      help='Multiple text queries')
    parser.add_argument('--classes', type=str, nargs='+',
                      help='Class names for zero-shot classification')
    parser.add_argument('--image_dir', type=str,
                      help='Directory with images for retrieval')
    parser.add_argument('--save_plot', type=str,
                      help='Path to save visualization')
    parser.add_argument('--top_k', type=int, default=5,
                      help='Number of top results to show')

    args = parser.parse_args()

    # Check model path
    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        return

    # Load CLIP model
    clip_model = CLIPInference(args.model_path)

    if args.mode == 'similarity':
        # Image-text similarity
        if not args.image or not args.texts:
            print("For similarity mode, provide --image and --texts")
            return

        print(f"Computing similarity between image and {len(args.texts)} texts...")
        similarities = clip_model.image_text_similarity(args.image, args.texts)

        print("\nResults:")
        for text, sim in zip(args.texts, similarities):
            print(f"'{text}': {sim:.4f}")

        # Visualize results
        visualize_results(args.image, args.texts, similarities, args.save_plot)

    elif args.mode == 'classification':
        # Zero-shot classification
        if not args.image or not args.classes:
            print("For classification mode, provide --image and --classes")
            return

        print(f"Performing zero-shot classification with {len(args.classes)} classes...")
        results = clip_model.zero_shot_classification(args.image, args.classes)

        print("\nTop predictions:")
        for i, result in enumerate(results[:args.top_k]):
            print(f"{i+1}. {result['class']}: {result['probability']:.4f} "
                  f"(similarity: {result['similarity']:.4f})")

        # Extract data for visualization
        classes = [r['class'] for r in results[:args.top_k]]
        similarities = [r['similarity'] for r in results[:args.top_k]]
        visualize_results(args.image, classes, similarities, args.save_plot)

    elif args.mode == 'retrieval':
        if args.text and args.image_dir:
            # Text-to-image retrieval
            image_files = [f for f in os.listdir(args.image_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image_paths = [os.path.join(args.image_dir, f) for f in image_files]

            if not image_paths:
                print(f"No images found in {args.image_dir}")
                return

            print(f"Searching for '{args.text}' in {len(image_paths)} images...")
            ranked_images, similarities = clip_model.text_image_retrieval(args.text, image_paths)

            print(f"\nTop {args.top_k} results:")
            for i, (image_path, sim) in enumerate(zip(ranked_images[:args.top_k], similarities[:args.top_k])):
                print(f"{i+1}. {os.path.basename(image_path)}: {sim:.4f}")

        elif args.image and args.texts:
            # Image-to-text retrieval
            print(f"Finding best text match for image from {len(args.texts)} options...")
            ranked_texts, similarities = clip_model.image_text_retrieval(args.image, args.texts)

            print(f"\nTop {args.top_k} results:")
            for i, (text, sim) in enumerate(zip(ranked_texts[:args.top_k], similarities[:args.top_k])):
                print(f"{i+1}. '{text}': {sim:.4f}")

        else:
            print("For retrieval mode, provide either --text and --image_dir, or --image and --texts")


def demo_inference(model_path):
    """Demonstration of CLIP inference capabilities"""
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    print("=== CLIP Inference Demo ===")

    # Load model
    clip_model = CLIPInference(model_path)

    # Check if we have demo images (from training)
    demo_image_dir = "data/images"
    if os.path.exists(demo_image_dir):
        image_files = [f for f in os.listdir(demo_image_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:5]

        if image_files:
            print("\n1. Image-Text Similarity Demo")
            test_image = os.path.join(demo_image_dir, image_files[0])
            test_texts = ["a red circle", "a blue square", "a green triangle",
                         "a yellow shape", "a purple object"]

            similarities = clip_model.image_text_similarity(test_image, test_texts)

            print(f"Image: {image_files[0]}")
            print("Text similarities:")
            for text, sim in zip(test_texts, similarities):
                print(f"  '{text}': {sim:.4f}")

            print("\n2. Zero-Shot Classification Demo")
            classes = ["red circle", "blue square", "green triangle", "yellow circle", "purple square"]
            results = clip_model.zero_shot_classification(test_image, classes)

            print("Top 3 predictions:")
            for i, result in enumerate(results[:3]):
                print(f"  {i+1}. {result['class']}: {result['probability']:.4f}")

            print("\n3. Text-to-Image Retrieval Demo")
            query_text = "a red shape"
            image_paths = [os.path.join(demo_image_dir, f) for f in image_files]
            ranked_images, similarities = clip_model.text_image_retrieval(query_text, image_paths)

            print(f"Query: '{query_text}'")
            print("Top 3 matching images:")
            for i, (image_path, sim) in enumerate(zip(ranked_images[:3], similarities[:3])):
                print(f"  {i+1}. {os.path.basename(image_path)}: {sim:.4f}")
    else:
        print("No demo images found. Train the model first to generate demo data.")


if __name__ == '__main__':
    import sys

    # If no arguments provided, run demo
    if len(sys.argv) == 1:
        model_path = "clip_model.pth"
        demo_inference(model_path)
    else:
        main()
