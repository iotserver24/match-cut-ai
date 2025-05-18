# Background Styles for Text Match Cut

This document explains the new background styles feature added to the Text Match Cut Video Generator.

## Available Background Styles

The application now supports the following background styles:

1. **Solid Color** (default)
   - A simple, solid color background
   - You can customize the background color using the color picker

2. **Newspaper**
   - Simulates a newspaper layout with columns and text blocks
   - Features a light gray background with column separators
   - Adds a subtle newsprint texture with specks

3. **Old Paper**
   - Creates an aged, vintage paper look
   - Features a yellowish-brown base color
   - Includes random coffee stains and edge darkening for a worn appearance

## How to Use

### Web Interface

1. Go to the Text Match Cut web interface
2. Fill in your highlighted text and other parameters
3. Under "Background Style", select your preferred style:
   - Solid Color
   - Newspaper
   - Old Paper
4. Click "Generate Video"

### API

When using the API, include the `background_style` parameter in your request:

```json
{
  "highlighted_text": "Your text here",
  "width": 1920,
  "height": 1080,
  "duration": 5,
  "background_style": "newspaper"  // Options: "solid", "newspaper", "old_paper"
}
```

## Technical Implementation

The background styles are implemented in `text_effect.py` using the following approach:

1. A new `create_background_texture()` function generates textures based on the selected style
2. The `create_text_image_frame()` function was updated to support different background styles
3. Each style has customized highlight rendering to match the style's aesthetic

## Example Images

Example background textures can be found in the `static/images/backgrounds` directory:

- `newspaper_example.jpg`: Example of the newspaper style
- `old_paper_example.jpg`: Example of the old paper style

## Adding Custom Styles

To add more background styles:

1. Update the `create_background_texture()` function in `text_effect.py`
2. Add your new style option to the dropdown in `templates/index.html`
3. Update the validation in `app.py` to include your new style
4. Update the API documentation in `API.md` and `templates/api_docs.html`

## Notes

- Background styles may affect the readability of text
- Choose text and highlight colors that work well with your selected background style
- The newspaper style works best with black text and light-colored highlights
- The old paper style works best with dark brown or black text 