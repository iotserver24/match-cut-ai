import os
from PIL import Image, ImageDraw, ImageFilter
import random

def create_newspaper_texture(width, height, output_path):
    """Create a newspaper-like texture and save it to the specified path."""
    # Draw a light gray background
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
    
    # Save the image
    img.save(output_path)
    print(f"Newspaper texture saved to {output_path}")

def create_old_paper_texture(width, height, output_path):
    """Create an old paper texture and save it to the specified path."""
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
    
    # Save the image
    img.save(output_path)
    print(f"Old paper texture saved to {output_path}")

def main():
    # Create the directory if it doesn't exist
    output_dir = "static/images/backgrounds"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate newspaper texture
    newspaper_path = os.path.join(output_dir, "newspaper_example.jpg")
    create_newspaper_texture(500, 500, newspaper_path)
    
    # Generate old paper texture
    old_paper_path = os.path.join(output_dir, "old_paper_example.jpg")
    create_old_paper_texture(500, 500, old_paper_path)
    
    print("Texture generation complete!")

if __name__ == "__main__":
    main() 