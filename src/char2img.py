from PIL import Image, ImageDraw, ImageFont
import numpy as np
from random import uniform, randint
import matplotlib.pyplot as plt

def default_font_size(text):
    num_chars = len(text)
    if num_chars == 1:
        font_size = 24
    elif num_chars == 2:
        font_size = 18
    elif num_chars == 3:
        font_size = 16
    else:
        raise ValueError("unsupported characters length")
    return font_size

def random_font_size(text):
    # Apply random scaling
    num_chars = len(text)
    if num_chars == 1:
        font_size = uniform(14, 26)
    elif num_chars == 2:
        font_size = uniform(14, 20)
    elif num_chars == 3:
        font_size = uniform(14, 16)
    else:
        raise ValueError("unsupported characters length")
    return font_size

def default_position(text_width, text_height):
    # Center the text
    x = (28 - text_width) / 2
    y = (28 - text_height) / 2
    return x, y

def random_position_translation(text_width, text_height):
    # Apply translation (only up and down)
    #x = uniform(0, 28 - text_width)
    x = text_width
    y = uniform(0, 28 - text_height)
    return x, y

def default_font():
    return "fonts/Inter_18pt-Bold.ttf"

def random_font():
    fonts = [
        "fonts/SourceSerif4_18pt-Bold.ttf",
        "fonts/Inter_18pt-Bold.ttf",
        "fonts/Questrial-Regular.ttf",
    ]
    selection = randint(0, len(fonts) - 1)
    return fonts[selection]

def generate_random_adversarial_image(text):
    # Create a 28x28 black image
    img = Image.new("L", (28, 28), color=0)
    draw = ImageDraw.Draw(img)

    random_scaling_prob = 0.7
    do_random_scaling = 1 if uniform(0, 1) < random_scaling_prob else 0
    if do_random_scaling:
        # Apply random scaling
        font_size = random_font_size(text)
    else:
        font_size = default_font_size(text)
    
    # Select a font
    font_file = default_font()
    font = ImageFont.truetype(font_file, size=font_size)
    
    # Get text size
    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
    
    random_translation_prob = 0.7
    do_random_translation = 1 if uniform(0, 1) < random_translation_prob else 0
    if do_random_translation:
        # Apply translation
        x, y = random_position_translation(text_width, text_height)
    else:
        x, y = default_position(text_width, text_height)

    random_rotation_prob = 0.7
    do_random_rotation = 1 if uniform(0, 1) < random_rotation_prob else 0
    if do_random_rotation:
        # Create a separate image for text to allow rotation
        text_img = Image.new("L", (text_width, text_height), color=0)
        text_draw = ImageDraw.Draw(text_img)
        text_draw.text((0, 0), text, fill=255, font=font)
        
        # Apply random rotation
        angle = randint(-10, 10)
        text_img = text_img.rotate(angle, expand=True, fillcolor=0)
        
        # Paste rotated text onto the main image
        img.paste(text_img, (int(x), int(y)), text_img)
    else:
        # Draw text directly onto the main image
        draw.text((x, y), text, fill=255, font=font)
    
    # Convert image to numpy array
    img_array = np.array(img)
    return img_array / 255

def generate_chars_image(text):
    # Create a 28x28 black image
    img = Image.new("L", (28, 28), color=0)
    draw = ImageDraw.Draw(img)

    # Font size
    font_size = default_font_size(text)
    
    #font = ImageFont.load_default(size=font_size)
    font_name = default_font()
    font = ImageFont.truetype(font_name, size=font_size)
    
    # Get text size and position
    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
    
    # Put text in the center
    x, y = default_position(text_width, text_height)
    
    # Draw text directly onto the main image
    draw.text((x, y), text, fill=255, font=font)

    # Convert image to numpy array
    img_array = np.array(img)
    return img_array / 255

def generate_chars_image_flip(text):
    # Create a 28x28 black image
    img = Image.new("L", (28, 28), color=0)
    draw = ImageDraw.Draw(img)

    # Font size
    font_size = default_font_size(text)
    
    #font = ImageFont.load_default(size=font_size)
    font_name = default_font()
    font = ImageFont.truetype(font_name, size=font_size)
    
    # Get text size and position
    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
    
    # Put text in the center
    x, y = default_position(text_width, text_height)
    
    # Draw text directly onto the main image
    draw.text((x, y), text, fill=255, font=font)

    # Flip the image horizontally
    img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Convert image to numpy array
    img_array = np.array(img)
    return img_array / 255

def generate_chars_stretch(text):
    # Create a 28x28 black image
    img = Image.new("L", (28, 28), color=0)
    draw = ImageDraw.Draw(img)

    # Font size
    font_size = default_font_size(text)
    
    #font = ImageFont.load_default(size=font_size)
    font_name = default_font()
    font = ImageFont.truetype(font_name, size=font_size)
    
    # Get text size and position
    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
    
    # Put text in the center
    x, y = default_position(text_width, text_height)
    
    # Draw text directly onto the main image
    draw.text((x, y), text, fill=255, font=font)

    # Stretch the width of the image (e.g., by 1.5x)
    stretch_factor = 1.5  # Adjust this value to control the stretching
    new_width = int(img.width * stretch_factor)
    img = img.resize((new_width, img.height), Image.Resampling.LANCZOS)

    # Crop or pad the image back to 28x28
    if new_width > 28:
        # Crop the image to 28x28
        left = (new_width - 28) // 2
        img = img.crop((left, 0, left + 28, 28))
    else:
        # Pad the image to 28x28
        padded_img = Image.new("L", (28, 28), color=0)
        left = (28 - new_width) // 2
        padded_img.paste(img, (left, 0))
        img = padded_img

    # Convert image to numpy array
    img_array = np.array(img)
    return img_array / 255

def show_img(array):
    # show numpy array as gray scale image using plt
    plt.figure(figsize=(1, 1))
    plt.imshow(array, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.show()