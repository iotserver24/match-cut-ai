import os
import random
import string
import sys
import time  # For retry delays
import requests  # For making HTTP requests

import matplotlib.font_manager as fm
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from moviepy import ImageSequenceClip

# --- AI Text Generation Settings ---
AI_GENERATION_ENABLED = True  # Always enabled since we're using free API
UNIQUE_TEXT_COUNT = 2  # Number of unique text snippets to generate/pre-pool
POLLINATIONS_API_URL = "https://text.pollinations.ai"

# Video settings
WIDTH = 1024
HEIGHT = 1024
FPS = 10
DURATION_SECONDS = 5
OUTPUT_FILENAME = "text_match_cut_video_v2.mp4"

# Text & Highlighting settings
HIGHLIGHTED_TEXT = "Mother of Dragons"
HIGHLIGHT_COLOR = "yellow"  # Pillow color name or hex code
TEXT_COLOR = "black"
BACKGROUND_COLOR = "white"
FONT_SIZE_RATIO = 0.05  # Adjusted slightly for multi-line potentially
MIN_LINES = 7  # Min number of text lines per frame
MAX_LINES = 10  # Max number of text lines per frame
VERTICAL_SPREAD_FACTOR = 1.5  # Multiplier for line height (1.0 = tight, 1.5 = looser)

# Effect settings
BLUR_TYPE = 'radial'  # Options: 'gaussian', 'radial'
BLUR_RADIUS = 4.0  # Gaussian blur radius, or the radius OUTSIDE which radial blur starts fading strongly
RADIAL_SHARPNESS_RADIUS_FACTOR = 0.3  # For 'radial': Percentage of min(W,H) to keep perfectly sharp around center

# Font settings
FONT_DIR = "fonts"  # Dedicated font folder recommended
MAX_FONT_RETRIES_PER_FRAME = 5
# Generate random words only using ASCII lowercase for fallback/disabled AI
FALLBACK_CHAR_SET = string.ascii_lowercase + " "

# Background style settings
BACKGROUND_STYLE = 'newspaper'  # Options: 'solid', 'newspaper', 'old_paper'
BACKGROUND_TEXTURE_DIR = "static/images/backgrounds"

# --- Helper Functions ---

def get_random_font(font_paths, exclude_list=None):
    """Selects a random font file path from the list, avoiding excluded ones."""
    available_fonts = list(set(font_paths) - set(exclude_list or []))
    if not available_fonts:
        # Fallback if all fonts failed or list is empty initially
        try:
            prop = fm.FontProperties(family='sans-serif')
            fallback = fm.findfont(prop, fallback_to_default=True)
            print(f"Warning: No usable fonts found from list/system. Using fallback: {fallback}")
            return fallback
        except Exception:
            print("ERROR: No fonts found and fallback failed. Cannot proceed.")
            return None
    return random.choice(available_fonts)


# Fallback random text generator
def generate_random_words(num_words):
    """Generates a string of random 'words' using only FALLBACK_CHAR_SET."""
    words = []
    for _ in range(num_words):
        length = random.randint(3, 8)
        word = ''.join(random.choice(FALLBACK_CHAR_SET.replace(" ", "")) for i in range(length))
        words.append(word)
    return " ".join(words)


def generate_random_text_snippet(highlighted_text, min_lines, max_lines):
    """Generates multiple lines of random text, ensuring MIN_LINES."""
    # Ensure we generate at least min_lines
    num_lines = random.randint(max(1, min_lines), max(min_lines, max_lines))  # Ensure at least min_lines generated
    highlight_line_index = random.randint(0, num_lines - 1)
    lines = []
    min_words_around = 2
    max_words_around = 6
    for i in range(num_lines):
        if i == highlight_line_index:
            words_before = generate_random_words(random.randint(min_words_around, max_words_around))
            words_after = generate_random_words(random.randint(min_words_around, max_words_around))
            lines.append(f"{words_before} {highlighted_text} {words_after}")
        else:
            lines.append(generate_random_words(random.randint(max_words_around, max_words_around * 2)))

    # Double-check final line count (should always pass with the adjusted randint)
    if len(lines) < min_lines:
        print(f"Warning: Random generator created only {len(lines)} lines (min: {min_lines}). This shouldn't happen.")
        return None, -1  # Treat as failure if check fails unexpectedly

    return lines, highlight_line_index


def generate_ai_text_snippet(highlighted_text, min_lines, max_lines):
    """Generates a text snippet using Pollinations API containing the highlighted text."""
    target_lines = random.randint(min_lines, max_lines)
    prompt = (
        f"Generate a text block of approximately {target_lines} distinct lines (aim for at least {min_lines}). "
        f"One of the lines MUST contain the exact phrase: '{highlighted_text}'. "
        f"The surrounding text should be thematically related to '{highlighted_text}' (e.g., fantasy, power, dragons, leadership). "
        f"Ensure the phrase '{highlighted_text}' fits naturally within its line. "
        f"Format the output ONLY as the text lines, each separated by a single newline character. Do not add any extra explanations or formatting."
    )

    try:
        # Make request to Pollinations API
        response = requests.get(f"{POLLINATIONS_API_URL}/{prompt}&model=mistral")
        response.raise_for_status()  # Raise exception for bad status codes
        
        # Get the text content from response
        content = response.text.strip()

        # Basic cleanup: remove potential empty lines
        lines = [line for line in content.split('\n') if line.strip()]

        # --- CRITICAL CHECK: Ensure minimum lines ---
        if len(lines) < min_lines:
            print(
                f"Warning: AI returned only {len(lines)} valid lines (minimum requested: {min_lines}). Retrying generation.")
            return None, -1  # Indicate failure due to insufficient lines

        # Find the highlight line
        highlight_line_index = -1
        for i, line in enumerate(lines):
            if highlighted_text in line:
                highlight_line_index = i
                break

        if highlight_line_index == -1:
            print(f"Warning: AI response did not contain the exact phrase '{highlighted_text}'.")
            return None, -1  # Indicate failure

        return lines, highlight_line_index
    except Exception as e:
        print(f"An unexpected error occurred during AI text generation: {e}")
        return None, -1  # Indicate failure


def create_radial_blur_mask(width, height, center_x, center_y, sharp_radius, fade_radius):
    """Creates a grayscale mask for radial blur (sharp center, fades out)."""
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(
        (center_x - sharp_radius, center_y - sharp_radius,
         center_x + sharp_radius, center_y + sharp_radius),
        fill=255
    )
    # Gaussian blur the sharp circle mask for a smooth falloff
    # Ensure fade radius is larger than sharp radius
    blur_amount = max(0.1, (fade_radius - sharp_radius) / 3.5)  # Adjusted divisor for smoothness
    mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_amount))
    return mask


def create_background_texture(width, height, style, bg_color):
    """Creates a textured background based on the specified style.
    
    Args:
        width (int): Width of the background image
        height (int): Height of the background image
        style (str): Background style ('solid', 'newspaper', 'old_paper', etc.)
        bg_color (str): Background color (for solid or tinting textures)
        
    Returns:
        PIL.Image: Background image with texture
    """
    print(f"Creating background texture: style={style}, color={bg_color}, size={width}x{height}")
    
    # Create base image with background color
    img = Image.new('RGB', (width, height), color=bg_color)
    
    if style == 'solid':
        print("Using solid background style")
        # Just return the solid color background
        return img
    
    # Check if texture directory exists
    if not os.path.isdir(BACKGROUND_TEXTURE_DIR):
        print(f"Background texture directory not found, creating: {BACKGROUND_TEXTURE_DIR}")
        os.makedirs(BACKGROUND_TEXTURE_DIR, exist_ok=True)
    
    # Generate newspaper texture
    if style == 'newspaper':
        print("Generating newspaper style background")
        
        # First try - use our simple fallback that's guaranteed to work
        try:
            print("Using simple newspaper background generator")
            return create_simple_newspaper_background(width, height, bg_color)
        except Exception as e:
            print(f"Simple newspaper background failed: {e}, trying next method")
        
        # Second try - use pre-generated example
        example_path = os.path.join(BACKGROUND_TEXTURE_DIR, "newspaper_example.jpg")
        print(f"Looking for newspaper texture at: {example_path}")
        if os.path.exists(example_path):
            try:
                print(f"Found newspaper texture at {example_path}")
                example = Image.open(example_path)
                # Resize to match requested dimensions
                example = example.resize((width, height), Image.LANCZOS)
                print(f"Successfully resized newspaper texture to {width}x{height}")
                return example
            except Exception as e:
                print(f"Error using example newspaper texture: {e}, generating full custom texture")
        else:
            print(f"Newspaper texture not found at {example_path}, generating full custom texture")
        
        # Generate a newspaper-like texture with columns and fake text
        draw = ImageDraw.Draw(img)
        
        # Draw a light gray background instead of solid color
        img = Image.new('RGB', (width, height), color="#f5f5f5")
        draw = ImageDraw.Draw(img)
        
        # Create newspaper columns
        col_count = random.randint(3, 4)  # 3 or 4 columns
        col_width = width // col_count
        col_padding = 20
        
        # Draw column separators
        for i in range(1, col_count):
            x = i * col_width
            draw.line([(x, 0), (x, height)], fill="#dddddd", width=1)
        
        # Draw fake text lines in columns
        line_height = 8
        for col in range(col_count):
            col_x = col * col_width + col_padding
            max_width = col_width - (2 * col_padding)
            
            # Draw column header
            header_y = random.randint(20, 40)
            draw.rectangle([(col_x, header_y), (col_x + max_width - 10, header_y + 20)], fill="#dddddd")
            
            # Draw fake text lines
            y = header_y + 40
            while y < height - 20:
                line_width = random.randint(int(max_width * 0.7), max_width)
                draw.rectangle([(col_x, y), (col_x + line_width, y + 4)], fill="#dddddd")
                y += line_height
        
        # Add some noise for newspaper print effect
        for _ in range(5000):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            draw.point((x, y), fill="#cccccc")
            
        return img
    
    # Generate old paper texture
    elif style == 'old_paper':
        print("Generating old paper style background")
        
        # First try - use our simple fallback that's guaranteed to work
        try:
            print("Using simple old paper background generator")
            return create_simple_old_paper_background(width, height, bg_color)
        except Exception as e:
            print(f"Simple old paper background failed: {e}, trying next method")
        
        # Second try - use pre-generated example
        example_path = os.path.join(BACKGROUND_TEXTURE_DIR, "old_paper_example.jpg")
        if os.path.exists(example_path):
            try:
                print(f"Found old paper texture at {example_path}")
                example = Image.open(example_path)
                # Resize to match requested dimensions
                example = example.resize((width, height), Image.LANCZOS)
                print(f"Successfully resized old paper texture to {width}x{height}")
                return example
            except Exception as e:
                print(f"Error using example old paper texture: {e}, generating full custom texture")
        else:
            print(f"Old paper texture not found at {example_path}, generating full custom texture")
        
        # Create a yellowish-brown base for old paper
        img = Image.new('RGB', (width, height), color="#f2e8c9")
        draw = ImageDraw.Draw(img)
        
        # Add paper grain texture
        for _ in range(width * height // 100):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            size = random.randint(1, 3)
            color_variation = random.randint(-15, 10)
            color = (210 + color_variation, 200 + color_variation, 165 + color_variation)
            draw.ellipse((x, y, x + size, y + size), fill=color)
        
        # Add some coffee stain-like spots
        for _ in range(random.randint(2, 5)):
            stain_x = random.randint(0, width - 1)
            stain_y = random.randint(0, height - 1)
            stain_size = random.randint(50, 150)
            stain_color = (random.randint(160, 180), random.randint(120, 140), random.randint(80, 100))
            stain_alpha = random.randint(30, 70)  # Transparency of the stain
            
            # Create a stain mask
            stain_mask = Image.new('L', (width, height), 0)
            stain_draw = ImageDraw.Draw(stain_mask)
            stain_draw.ellipse(
                (stain_x - stain_size, stain_y - stain_size, 
                 stain_x + stain_size, stain_y + stain_size), 
                fill=stain_alpha
            )
            
            # Blur the stain for a more natural look
            stain_mask = stain_mask.filter(ImageFilter.GaussianBlur(radius=stain_size//4))
            
            # Create a stain overlay
            stain_overlay = Image.new('RGB', (width, height), stain_color)
            
            # Composite the stain onto the paper
            img = Image.composite(stain_overlay, img, stain_mask)
        
        # Add edge darkening for worn look
        edge_mask = Image.new('L', (width, height), 255)
        edge_draw = ImageDraw.Draw(edge_mask)
        edge_width = min(width, height) // 10
        
        # Draw lighter center, darker edges
        edge_draw.rectangle(
            (edge_width, edge_width, width - edge_width, height - edge_width),
            fill=0
        )
        edge_mask = edge_mask.filter(ImageFilter.GaussianBlur(radius=edge_width//2))
        
        # Create edge overlay
        edge_overlay = Image.new('RGB', (width, height), (160, 140, 100))
        
        # Apply edge effect
        img = Image.composite(edge_overlay, img, edge_mask)
        
        return img
    
    # Add more style options here
    else:
        # Default to solid background if style not recognized
        print(f"WARNING: Unknown background style '{style}'. Using solid background.")
        return img


def create_text_image_frame(width, height, text_lines, highlight_line_index, highlighted_text,
                            font_path, font_size, text_color, bg_color, highlight_color,
                            blur_type, blur_radius, radial_sharp_radius_factor, vertical_spread_factor,
                            background_style='solid'):
    """Creates a single frame image with centered highlight and multi-line text."""
    print(f"\nCreating frame with background_style='{background_style}', blur_type='{blur_type}'")

    # --- Font Loading ---
    try:
        font = ImageFont.truetype(font_path, font_size)
        bold_font = font  # Start with regular as fallback
        # Simple bold variant check (can be improved)
        common_bold_suffixes = ["bd.ttf", "-Bold.ttf", "b.ttf", "_Bold.ttf", " Bold.ttf"]
        base_name, ext = os.path.splitext(font_path)
        for suffix in common_bold_suffixes:
            potential_bold_path = base_name.replace("Regular", "").replace("regular",
                                                                           "") + suffix  # Try removing 'Regular' too
            if os.path.exists(potential_bold_path):
                try:
                    bold_font = ImageFont.truetype(potential_bold_path, font_size)
                    # print(f"    Using bold variant: {os.path.basename(potential_bold_path)}") # Debug
                    break  # Use the first one found
                except IOError:
                    continue  # Try next suffix if loading fails
            # Check without removing Regular if first checks failed
            potential_bold_path = base_name + suffix
            if os.path.exists(potential_bold_path):
                try:
                    bold_font = ImageFont.truetype(potential_bold_path, font_size)
                    # print(f"    Using bold variant: {os.path.basename(potential_bold_path)}") # Debug
                    break
                except IOError:
                    continue

    except IOError as e:
        raise FontLoadError(f"Failed to load font: {font_path}") from e
    except Exception as e:  # Catch other potential font loading issues
        raise FontLoadError(f"Unexpected error loading font {font_path}: {e}") from e

    # --- Calculations ---
    try:
        # Line height using getmetrics()
        try:
             ascent, descent = font.getmetrics()
             metric_height = ascent + abs(descent)
             line_height = int(metric_height * vertical_spread_factor)
        except AttributeError:
             bbox_line_test = font.getbbox("Ay", anchor="lt")
             line_height = int((bbox_line_test[3] - bbox_line_test[1]) * vertical_spread_factor)
        if line_height <= font_size * 0.8:
            line_height = int(font_size * 1.2 * vertical_spread_factor)

        # BOLD font metrics for final highlight placement
        highlight_width_bold = bold_font.getlength(highlighted_text)
        highlight_bbox_h = bold_font.getbbox(highlighted_text, anchor="lt")
        highlight_height_bold = highlight_bbox_h[3] - highlight_bbox_h[1]
        if highlight_width_bold <= 0 or highlight_height_bold <= 0:
             highlight_height_bold = int(font_size * 1.1)
             if highlight_width_bold <=0: highlight_width_bold = len(highlighted_text) * font_size * 0.6

        # Target position for the TOP-LEFT of the final BOLD highlight text (CENTERED)
        highlight_target_x = (width - highlight_width_bold) / 2
        highlight_target_y = (height - highlight_height_bold) / 2

        # Block start Y calculated relative to the centered highlight's top
        block_start_y = highlight_target_y - (highlight_line_index * line_height)

        # Get Prefix and Suffix for background alignment
        highlight_line_full_text = text_lines[highlight_line_index]
        prefix_text = ""
        suffix_text = "" # Also get suffix now
        highlight_found_in_line = False
        try:
            start_index = highlight_line_full_text.index(highlighted_text)
            end_index = start_index + len(highlighted_text)
            prefix_text = highlight_line_full_text[:start_index]
            suffix_text = highlight_line_full_text[end_index:]
            highlight_found_in_line = True
        except ValueError: pass # Treat line normally if not found

        # Measure Prefix Width using REGULAR font (for background positioning)
        prefix_width_regular = font.getlength(prefix_text)
        # Calculate the required starting X for the background highlight line string
        # This is the coordinate used for drawing the *full string* in the background
        bg_highlight_line_start_x = highlight_target_x - prefix_width_regular

    except AttributeError: raise FontDrawError(f"Font lacks methods.")
    except Exception as e: raise FontDrawError(f"Measurement fail: {e}") from e

    # --- Create background with texture ---
    print(f"Calling create_background_texture with style='{background_style}'")
    img_base = create_background_texture(width, height, background_style, bg_color)
    draw_base = ImageDraw.Draw(img_base)
    
    # --- Base Image Drawing (Draw FULL lines, use offset for HL line) ---
    try:
        current_y = block_start_y
        for i, line in enumerate(text_lines):
            line_x = 0.0
            if i == highlight_line_index and highlight_found_in_line:
                line_x = bg_highlight_line_start_x
            else:
                line_width = font.getlength(line)
                line_x = (width - line_width) / 2
            draw_base.text((line_x, current_y), line, font=font, fill=text_color, anchor="lt")
            current_y += line_height
    except Exception as e: raise FontDrawError(f"Base draw fail: {e}") from e

    # --- Apply Blur (with padding for Gaussian to avoid edge clipping) ---
    img_blurred = None # Initialize
    padding_for_blur = int(blur_radius * 3)

    if blur_type == 'gaussian' and blur_radius > 0:
        print(f"Applying gaussian blur with radius={blur_radius}")
        try:
            # Create larger canvas
            padded_width = width + 2 * padding_for_blur
            padded_height = height + 2 * padding_for_blur
            
            # Create padded background texture
            img_padded = create_background_texture(padded_width, padded_height, background_style, bg_color)
            
            # Paste original centered onto padded canvas
            img_padded.paste(img_base, (padding_for_blur, padding_for_blur))
            # Blur the padded image
            img_padded_blurred = img_padded.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            # Crop the center back to original size
            img_blurred = img_padded_blurred.crop((padding_for_blur, padding_for_blur,
                                                  padding_for_blur + width, padding_for_blur + height))
        except Exception as e:
            print(f"Error during padded Gaussian blur: {e}. Falling back to direct blur.")
            img_blurred = img_base.filter(ImageFilter.GaussianBlur(radius=blur_radius)) # Fallback


    elif blur_type == 'radial' and blur_radius > 0:
        print(f"Applying radial blur with radius={blur_radius}")
        # For radial, we need img_sharp. Let's try drawing it *in parts* for reliability
        # as the padded blur trick doesn't apply directly here.
        img_sharp = create_background_texture(width, height, background_style, bg_color)
        draw_sharp = ImageDraw.Draw(img_sharp)
        try:
            current_y = block_start_y
            for i, line in enumerate(text_lines):
                if i == highlight_line_index and highlight_found_in_line:
                    # --- Draw Sharp Highlight Line in Parts ---
                    # Calculate positions relative to the *final* centered highlight target
                    prefix_x = highlight_target_x - prefix_width_regular
                    # Use REGULAR font for the sharp layer (it's just for the mask)
                    draw_sharp.text((prefix_x, current_y), prefix_text, font=font, fill=text_color, anchor="lt")
                    # Highlight part itself starts at highlight_target_x
                    highlight_width_regular = font.getlength(highlighted_text) # Width in regular font
                    draw_sharp.text((highlight_target_x, current_y), highlighted_text, font=font, fill=text_color, anchor="lt")
                    # Suffix starts after the regular highlight width
                    suffix_x = highlight_target_x + highlight_width_regular
                    draw_sharp.text((suffix_x, current_y), suffix_text, font=font, fill=text_color, anchor="lt")
                else:
                    # Draw non-highlight lines centered normally
                    line_width = font.getlength(line)
                    line_x = (width - line_width) / 2
                    draw_sharp.text((line_x, current_y), line, font=font, fill=text_color, anchor="lt")
                current_y += line_height
        except Exception as e:
             raise FontDrawError(f"Failed sharp text draw (parts): {e}") from e

        # Composite blurred base and sharp center
        # Base image (img_base) still uses the offset drawing method for full line
        img_fully_blurred = img_base.filter(ImageFilter.GaussianBlur(radius=blur_radius * 1.5))
        sharp_center_radius = min(width, height) * radial_sharp_radius_factor
        fade_radius = sharp_center_radius + max(width, height) * 0.15
        mask = create_radial_blur_mask(width, height, width / 2, height / 2, sharp_center_radius, fade_radius)
        img_blurred = Image.composite(img_sharp, img_fully_blurred, mask)

    else: # No blur
        print("No blur applied")
        img_blurred = img_base.copy()


    # --- Final Image: Draw ONLY Highlight Rectangle & Centered BOLD Text ---
    final_img = img_blurred # Start with the blurred/composited image
    draw_final = ImageDraw.Draw(final_img)
    try:
        # 1. Draw highlight rectangle (centered using bold metrics)
        padding = font_size * 0.10
        
        # Customize highlight style based on background_style
        if background_style == 'newspaper':
            print("Applying newspaper-style highlight")
            # For newspaper style, use a more traditional highlight (like a marker)
            draw_final.rectangle(
                [
                    (highlight_target_x - padding, highlight_target_y - padding),
                    (highlight_target_x + highlight_width_bold + padding, highlight_target_y + highlight_height_bold + padding)
                ],
                fill=highlight_color,
                outline="#000000",
                width=1
            )
        elif background_style == 'old_paper':
            print("Applying old-paper-style highlight")
            # For old paper, use a more subtle highlight with slight transparency
            # Create a semi-transparent overlay for the highlight
            highlight_overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            highlight_draw = ImageDraw.Draw(highlight_overlay)
            highlight_draw.rectangle(
                [
                    (highlight_target_x - padding, highlight_target_y - padding),
                    (highlight_target_x + highlight_width_bold + padding, highlight_target_y + highlight_height_bold + padding)
                ],
                fill=highlight_color
            )
            # Convert highlight_overlay to RGB mode to match final_img
            highlight_overlay = highlight_overlay.convert('RGB')
            
            # Create a mask for the highlight area
            highlight_mask = Image.new('L', (width, height), 0)
            highlight_mask_draw = ImageDraw.Draw(highlight_mask)
            highlight_mask_draw.rectangle(
                [
                    (highlight_target_x - padding, highlight_target_y - padding),
                    (highlight_target_x + highlight_width_bold + padding, highlight_target_y + highlight_height_bold + padding)
                ],
                fill=180  # Semi-transparent (0-255)
            )
            
            # Apply the highlight with the mask
            final_img = Image.composite(highlight_overlay, final_img, highlight_mask)
            draw_final = ImageDraw.Draw(final_img)
        else:
            print("Applying default-style highlight")
            # Default highlight style
            draw_final.rectangle(
                [
                    (highlight_target_x - padding, highlight_target_y - padding),
                    (highlight_target_x + highlight_width_bold + padding, highlight_target_y + highlight_height_bold + padding)
                ],
                fill=highlight_color
            )

        # 2. Draw ONLY the SHARP highlight text using BOLD font at the *perfectly centered* position
        draw_final.text(
            (highlight_target_x, highlight_target_y),
            highlighted_text,
            font=bold_font, # Use BOLD font
            fill=text_color,
            anchor="lt"
        )
        # *** No prefix/suffix drawing here ***

    except Exception as e:
         raise FontDrawError(f"Failed final highlight draw: {e}") from e

    print(f"Frame created successfully with background_style='{background_style}'")
    return final_img


# Custom Exceptions for font errors
class FontLoadError(Exception): pass
class FontDrawError(Exception): pass


def create_simple_newspaper_background(width, height, bg_color):
    """Creates a simple newspaper-style background that will reliably work.
    This is a fallback implementation for when the texture-based approach fails."""
    print("*** Using SIMPLE NEWSPAPER BACKGROUND as a fallback ***")
    
    # Create base with light gray
    img = Image.new('RGB', (width, height), color="#f5f5f5")
    draw = ImageDraw.Draw(img)
    
    # Add column lines
    columns = 4
    column_width = width // columns
    
    for i in range(1, columns):
        x = i * column_width
        draw.line([(x, 0), (x, height)], fill="#cccccc", width=2)
    
    # Add some random gray rectangles to simulate text
    for i in range(500):
        x = random.randint(0, width - 50)
        y = random.randint(0, height - 10)
        w = random.randint(20, 100)
        h = random.randint(5, 8)
        draw.rectangle([(x, y), (x + w, y + h)], fill="#dddddd")
    
    # Add some noise specks
    for i in range(5000):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        draw.point((x, y), fill="#bbbbbb")
    
    return img


def create_simple_old_paper_background(width, height, bg_color):
    """Creates a simple old paper background that will reliably work.
    This is a fallback implementation for when the texture-based approach fails."""
    print("*** Using SIMPLE OLD PAPER BACKGROUND as a fallback ***")
    
    # Create base with sepia color
    base_color = "#f2e8c9"  # Yellowish brown
    img = Image.new('RGB', (width, height), color=base_color)
    draw = ImageDraw.Draw(img)
    
    # Add grain effect
    for i in range(width * height // 500):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        size = random.randint(1, 3)
        color_var = random.randint(-20, 10)
        color = (210 + color_var, 200 + color_var, 165 + color_var)
        draw.ellipse((x, y, x + size, y + size), fill=color)
    
    # Add dark edges
    edge_width = min(width, height) // 10
    
    # Draw four edge rectangles
    # Top edge
    draw.rectangle([(0, 0), (width, edge_width)], 
                   fill=(180, 160, 120))
    # Bottom edge
    draw.rectangle([(0, height - edge_width), (width, height)], 
                   fill=(180, 160, 120))
    # Left edge
    draw.rectangle([(0, 0), (edge_width, height)], 
                   fill=(180, 160, 120))
    # Right edge
    draw.rectangle([(width - edge_width, 0), (width, height)], 
                   fill=(180, 160, 120))
    
    # Add a couple of coffee stains
    for _ in range(3):
        stain_x = random.randint(edge_width, width - edge_width)
        stain_y = random.randint(edge_width, height - edge_width)
        stain_size = random.randint(30, 80)
        draw.ellipse(
            (stain_x - stain_size, stain_y - stain_size,
             stain_x + stain_size, stain_y + stain_size),
            fill=(180, 150, 100)
        )
    
    return img


def main():
    print("Starting video generation...")

    # --- Font Discovery ---
    font_paths = []
    # (Font discovery code remains the same)
    if FONT_DIR and os.path.isdir(FONT_DIR):
        print(f"Looking for fonts in specified directory: {FONT_DIR}")
        for filename in os.listdir(FONT_DIR):
            if filename.lower().endswith((".ttf", ".otf")):
                font_paths.append(os.path.join(FONT_DIR, filename))
    else:
        print("FONT_DIR not specified or invalid, searching system fonts...")
        try:
            font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        except Exception as e:
            print(f"Error finding system fonts: {e}")

    if not font_paths:
        print("ERROR: No fonts found. Please install fonts or specify a valid FONT_DIR.")
        sys.exit(1)
    print(f"Found {len(font_paths)} potential fonts.")

    # --- Pre-generate Text Snippets ---
    text_snippets_pool = []
    print(f"Pre-generating {UNIQUE_TEXT_COUNT} unique text snippets...")

    generated_count = 0
    attempts = 0
    max_attempts = UNIQUE_TEXT_COUNT * 3  # Allow for some failures

    while generated_count < UNIQUE_TEXT_COUNT and attempts < max_attempts:
        attempts += 1
        print(f"  Generating snippet {generated_count + 1}/{UNIQUE_TEXT_COUNT} (Attempt {attempts})...")
        lines, hl_index = generate_ai_text_snippet(HIGHLIGHTED_TEXT, MIN_LINES, MAX_LINES)
        if lines and hl_index != -1:
            text_snippets_pool.append({"lines": lines, "highlight_index": hl_index})
            generated_count += 1
        else:
            print("    Failed to generate valid snippet, trying again.")
            time.sleep(1)  # Wait a bit before retrying on failure

    if generated_count < UNIQUE_TEXT_COUNT:
        print(
            f"Warning: Only generated {generated_count}/{UNIQUE_TEXT_COUNT} unique AI snippets after {max_attempts} attempts.")
        if generated_count == 0:
            print("ERROR: Failed to generate any AI text snippets. Cannot proceed with AI enabled.")
            sys.exit(1)

    # --- Calculate Other Parameters ---
    total_frames = int(FPS * DURATION_SECONDS)
    font_size = int(HEIGHT * FONT_SIZE_RATIO)
    print(f"\nVideo Settings: {WIDTH}x{HEIGHT} @ {FPS}fps, {DURATION_SECONDS}s ({total_frames} frames)")
    print(f"Text Settings: Highlight='{HIGHLIGHTED_TEXT}', Size={font_size}px, Spread={VERTICAL_SPREAD_FACTOR}")
    print(f"Effect Settings: BlurType='{BLUR_TYPE}', BlurRadius={BLUR_RADIUS}, HighlightColor='{HIGHLIGHT_COLOR}'")
    print(f"Using {'AI' if AI_GENERATION_ENABLED else 'Random'} Text Pool Size: {len(text_snippets_pool)}")

    # --- Generate Frames ---
    frames = []
    failed_fonts = set()
    print("\nGenerating frames...")
    frame_num = 0
    while frame_num < total_frames:
        print(f"  Attempting Frame {frame_num + 1}/{total_frames}")

        # Select a text snippet from the pool
        snippet = random.choice(text_snippets_pool)
        current_lines = snippet["lines"]
        highlight_idx = snippet["highlight_index"]

        font_retries = 0
        frame_generated = False
        while font_retries < MAX_FONT_RETRIES_PER_FRAME:
            current_font_path = get_random_font(font_paths, exclude_list=failed_fonts)
            if current_font_path is None:
                print("ERROR: Exhausted all available fonts or fallback failed. Stopping.")
                sys.exit(1)  # Exit if no fonts work

            try:
                img = create_text_image_frame(
                    WIDTH, HEIGHT,
                    current_lines, highlight_idx, HIGHLIGHTED_TEXT,
                    current_font_path, font_size,
                    TEXT_COLOR, BACKGROUND_COLOR, HIGHLIGHT_COLOR,
                    BLUR_TYPE, BLUR_RADIUS, RADIAL_SHARPNESS_RADIUS_FACTOR,
                    VERTICAL_SPREAD_FACTOR, BACKGROUND_STYLE
                )

                frame_np = np.array(img)
                frames.append(frame_np)
                frame_generated = True
                # print(f"    Frame {frame_num + 1} generated with font: {os.path.basename(current_font_path)}") # Less verbose
                break  # Success, next frame

            except (FontLoadError, FontDrawError) as e:
                # print(f"    Warning: Font '{os.path.basename(current_font_path)}' failed ({e}). Retrying frame.") # Less verbose
                failed_fonts.add(current_font_path)
                font_retries += 1
                # time.sleep(0.05) # Optional small delay
            except Exception as e:
                print(
                    f"    ERROR: Unexpected error generating frame with font {os.path.basename(current_font_path)}: {e}")
                failed_fonts.add(current_font_path)
                font_retries += 1
                # time.sleep(0.05)

        if not frame_generated:
            print(
                f"ERROR: Failed to generate Frame {frame_num + 1} after {MAX_FONT_RETRIES_PER_FRAME} font attempts. Stopping video generation.")
            break  # Stop if a frame repeatedly fails

        frame_num += 1

    # --- Create Video ---
    if not frames:
        print("ERROR: No frames were generated. Cannot create video.")
        sys.exit(1)

    if len(frames) < total_frames:
        print(f"Warning: Only {len(frames)}/{total_frames} frames were generated due to errors. Video will be shorter.")

    print("\nCompiling video...")
    try:
        clip = ImageSequenceClip(frames, fps=FPS)
        # Explicitly use 'libx264' for broad compatibility, 'bar' for progress
        clip.write_videofile(OUTPUT_FILENAME, codec='libx264', fps=FPS, logger='bar')
        print(f"\nVideo saved successfully as '{OUTPUT_FILENAME}'")
    except Exception as e:
        print(f"\nError during video writing: {e}")
        print("Check FFmpeg installation and codec support ('libx264').")

    if failed_fonts:
        print("\nFonts that caused errors (check character support/validity):")
        for ff in sorted(list(failed_fonts)):  # Sort for cleaner output
            print(f" - {os.path.basename(ff)}")

    print("\nScript finished.")


# --- Main Script Logic ---
if __name__ == "__main__":
    if not AI_GENERATION_ENABLED:
        print("NOTE: Mistral AI library not found or AI_GENERATION_ENABLED is False.")
        print("      The script will use RANDOM text generation.")
    elif not POLLINATIONS_API_URL:
        print("WARNING: POLLINATIONS_API_URL environment variable not found.")
        print("         AI text generation will fail if enabled.")
        # The main function will catch this and exit if AI is enabled.
    main()
