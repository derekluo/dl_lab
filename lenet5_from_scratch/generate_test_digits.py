import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def generate_digit_image(digit, size=(32, 32), padding=4):
    """Generate a white digit on black background image."""
    # Create a black background
    image = Image.new('L', size, color=0)
    draw = ImageDraw.Draw(image)

    # Try to load a font, fallback to default if not found
    try:
        # Adjust font size to fit the image with padding
        font_size = size[0] - 2 * padding
        font = ImageFont.truetype("Arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Get the size of the digit when rendered
    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Calculate position to center the digit
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2

    # Draw the white digit
    draw.text((x, y), text, fill=255, font=font)

    return image

def generate_test_set():
    """Generate test images for all digits 0-9."""
    # Create directory if it doesn't exist
    if not os.path.exists('test_digits'):
        os.makedirs('test_digits')

    # Generate an image for each digit
    for digit in range(10):
        image = generate_digit_image(digit)
        image.save(f'test_digits/digit_{digit}.png')
        print(f'Generated test_digits/digit_{digit}.png')

if __name__ == '__main__':
    generate_test_set()