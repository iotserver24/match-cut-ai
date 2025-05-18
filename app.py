import os
import random
import string
import time
import traceback  # For detailed error logging
import uuid  # For unique filenames
import requests  # For making HTTP requests
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import threading  # Add threading support
import redis  # For Redis integration
import json

import matplotlib.font_manager as fm
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from dotenv import load_dotenv
from flask import Flask, request, render_template, send_from_directory, url_for, flash, redirect, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from moviepy import ImageSequenceClip  # Use .editor for newer moviepy versions
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()

# --- Redis Setup ---
try:
    redis_client = redis.Redis(
        host='cosmic-titmouse-15186.upstash.io',
        port=6379,
        password='ATtSAAIjcDEyN2QxYTg0Y2U0ZjI0NWM2YTkyZTVkNmQyZTQwMTFlYnAxMA',
        ssl=True
    )
    # Test the connection
    redis_client.ping()
    print("Successfully connected to Redis")
except Exception as e:
    print(f"Error connecting to Redis: {e}")
    redis_client = None

# --- Catbox API Settings ---
CATBOX_API_URL = "https://catbox.moe/user/api.php"

# --- AI Text Generation Settings ---
POLLINATIONS_API_URL = "https://text.pollinations.ai"

# --- Flask App Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24) # Needed for flash messages (optional but good practice)
app.config['UPLOAD_FOLDER'] = 'output'
app.config['FONT_DIR'] = 'fonts'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['MAX_VIDEOS_PER_DAY'] = 10  # Per IP
app.config['CLEANUP_MINUTES'] = 10  # Delete videos after 10 minutes

# Rate limiter setup
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Ensure output and font directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['FONT_DIR'], exist_ok=True)

# --- Configuration Parameters ---

# Video settings
WIDTH = 1024
HEIGHT = 1024
FPS = 10
DURATION_SECONDS = 5

# Text & Highlighting settings
HIGHLIGHTED_TEXT = "Mother of Dragons"
HIGHLIGHT_COLOR = "yellow"  # Pillow color name or hex code
TEXT_COLOR = "black"
BACKGROUND_COLOR = "white"
FONT_SIZE_RATIO = 0.05  # Adjusted slightly for multi-line potentially
MIN_LINES = 7  # Min number of text lines per frame
MAX_LINES = 10  # Max number of text lines per frame
VERTICAL_SPREAD_FACTOR = 1.5  # Multiplier for line height (1.0 = tight, 1.5 = looser)

# AI Text Generation Settings
AI_GENERATION_ENABLED = True  # Always available now
UNIQUE_TEXT_COUNT = 2  # Number of unique text snippets to generate/pre-pool
MISTRAL_MODEL = "mistral-large-latest"  # Or choose another suitable model
# !! IMPORTANT: Load API Key securely !!
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

# Effect settings
BLUR_TYPE = 'radial'  # Options: 'gaussian', 'radial'
BLUR_RADIUS = 4.0  # Gaussian blur radius, or the radius OUTSIDE which radial blur starts fading strongly
RADIAL_SHARPNESS_RADIUS_FACTOR = 0.3  # For 'radial': Percentage of min(W,H) to keep perfectly sharp around center

# Font settings
FONT_DIR = "fonts"  # Dedicated font folder recommended
MAX_FONT_RETRIES_PER_FRAME = 5
# Generate random words only using ASCII lowercase for fallback/disabled AI
FALLBACK_CHAR_SET = string.ascii_lowercase + " "

# --- Helper Functions (Mostly unchanged from original script) ---

class FontLoadError(Exception): pass
class FontDrawError(Exception): pass

def get_random_font(font_paths, exclude_list=None):
    """Selects a random font file path from the list, avoiding excluded ones."""
    available_fonts = list(set(font_paths) - set(exclude_list or []))
    if not available_fonts:
        try:
            # More robust fallback finding sans-serif
            prop = fm.FontProperties(family='sans-serif')
            fallback_path = fm.findfont(prop, fallback_to_default=True)
            if fallback_path:
                 print(f"Warning: No usable fonts found from list/system. Using fallback: {fallback_path}")
                 return fallback_path
            else:
                 # If even matplotlib fallback fails (unlikely but possible)
                 print("ERROR: No fonts found in specified dir, system, or fallback. Cannot proceed.")
                 return None
        except Exception as e:
            print(f"ERROR: Font fallback mechanism failed: {e}. Cannot proceed.")
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

# Pollinations API Text Generation Function
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


def create_text_image_frame(width, height, text_lines, highlight_line_index, highlighted_text,
                            font_path, font_size, text_color, bg_color, highlight_color,
                            blur_type, blur_radius, radial_sharp_radius_factor, vertical_spread_factor,
                            background_style):
    """Creates a single frame image with centered highlight and multi-line text."""
    print(f"APP.PY: Called create_text_image_frame with background_style='{background_style}'")
    
    # Now import and call the actual implementation from text_effect
    from text_effect import create_text_image_frame as text_effect_create_frame
    
    # Call the implementation in text_effect.py
    return text_effect_create_frame(
        width, height, text_lines, highlight_line_index, highlighted_text,
        font_path, font_size, text_color, bg_color, highlight_color,
        blur_type, blur_radius, radial_sharp_radius_factor, vertical_spread_factor,
        background_style
    )


# --- Core Video Generation Logic (Adapted from main) ---
def generate_video(params):
    """Generates the video based on input parameters."""

    # Unpack parameters from the dictionary passed by the Flask route
    width = params['width']
    height = params['height']
    fps = params['fps']
    duration_seconds = params['duration']
    highlighted_text = params['highlighted_text']
    highlight_color = params['highlight_color']
    text_color = params['text_color']
    background_color = params['background_color']
    blur_type = params['blur_type']
    blur_radius = params['blur_radius']
    ai_enabled = params['ai_enabled']
    background_style = params.get('background_style', 'solid')  # Default to solid if not provided
    font_dir = app.config['FONT_DIR'] # Use font dir from Flask config

    # Add explicit debug prints about background_style
    print("=" * 50)
    print(f"BACKGROUND STYLE DEBUG: Using style '{background_style}'")
    print("Parameters for video generation:")
    print(f"- width: {width}")
    print(f"- height: {height}")
    print(f"- fps: {fps}")
    print(f"- duration: {duration_seconds}")
    print(f"- highlighted_text: {highlighted_text}")
    print(f"- highlight_color: {highlight_color}")
    print(f"- text_color: {text_color}")
    print(f"- background_color: {background_color}")
    print(f"- blur_type: {blur_type}")
    print(f"- blur_radius: {blur_radius}")
    print(f"- ai_enabled: {ai_enabled}")
    print(f"- background_style: {background_style}")
    print(f"- font_dir: {font_dir}")
    print("=" * 50)

    # Hardcoded or derived settings from original script
    font_size_ratio = 0.05 # Could be made a parameter
    min_lines = 7
    max_lines = 10
    vertical_spread_factor = 1.5
    radial_sharp_radius_factor = 0.3
    unique_text_count = 2 # Generate a couple of options per request
    mistral_model = "mistral-large-latest" # Could be param

    print(f"Starting video generation with params: {params}")

    # --- Font Discovery ---
    font_paths = []
    if font_dir and os.path.isdir(font_dir):
        print(f"Looking for fonts in specified directory: {font_dir}")
        for filename in os.listdir(font_dir):
            if filename.lower().endswith((".ttf", ".otf")):
                font_paths.append(os.path.join(font_dir, filename))
    else:
        print("FONT_DIR not specified or invalid, searching system fonts...")
        try:
            # Limit search to common locations if possible, or search all
            font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
            # font_paths.extend(fm.findSystemFonts(fontpaths=None, fontext='otf'))
        except Exception as e:
            print(f"Error finding system fonts: {e}")
            font_paths = []

    # --- Video Generation ---
    num_frames = int(duration_seconds * fps)
    font_size = int(min(width, height) * font_size_ratio)
    frames = []
    
    # Failed font tracking to avoid repeated failures
    failed_fonts = set()
    
    # Ensure we have usable fonts
    if not font_paths:
        return None, "No usable fonts found."

    # --- Generate Pool of Text Snippets First ---
    text_snippets = []
    highlight_line_indices = []
    
    # Use AI generation if enabled, function available, and API working
    if ai_enabled and "generate_ai_text_snippet" in globals():
        print("Generating AI text...")
        # Try AI generation with retry
        for _ in range(unique_text_count):
            for attempt in range(3):  # 3 tries per snippet
                lines, hl_index = generate_ai_text_snippet(highlighted_text, min_lines, max_lines)
                if lines and hl_index >= 0:
                    text_snippets.append(lines)
                    highlight_line_indices.append(hl_index)
                    break
                time.sleep(1)  # Give API a sec before retrying
    
    # If AI failed or not enough snippets, use fallback
    while len(text_snippets) < unique_text_count:
        lines, hl_index = generate_random_text_snippet(highlighted_text, min_lines, max_lines)
        text_snippets.append(lines)
        highlight_line_indices.append(hl_index)

    if not text_snippets:
        return None, "Failed to generate any valid text snippets."

    # --- Generate Frames Using Text Snippets ---
    unique_fonts_per_snippet = {}  # track which fonts to use with which snippet
    
    for snippet_idx, (text_lines, hl_idx) in enumerate(zip(text_snippets, highlight_line_indices)):
        unique_fonts_per_snippet[snippet_idx] = []
        # Decide on a few random fonts for this snippet
        usable_fonts = [f for f in font_paths if f not in failed_fonts]
        if not usable_fonts:
            return None, "No usable fonts after filtering failed ones."
        font_count = min(5, len(usable_fonts))
        unique_fonts_per_snippet[snippet_idx] = random.sample(usable_fonts, font_count)
    
    print(f"Generating {num_frames} frames...")
    for frame_index in range(num_frames):
        # Select text snippet pattern to use (rotate through available ones)
        snippet_idx = frame_index % len(text_snippets)
        text_lines = text_snippets[snippet_idx]
        highlight_line_index = highlight_line_indices[snippet_idx]
        
        # Try different fonts if needed
        frame_generated = False
        
        for font_try in range(MAX_FONT_RETRIES_PER_FRAME):
            try:
                if not unique_fonts_per_snippet[snippet_idx]:
                    # If all fonts for this snippet failed, try all remaining usable fonts
                    font_path = get_random_font(font_paths, failed_fonts)
                else:
                    # Use one of the pre-selected fonts for this snippet
                    font_path = random.choice(unique_fonts_per_snippet[snippet_idx])
                
                if not font_path:
                    print("Error: get_random_font returned None, no fonts available.")
                    return None, "No usable fonts remaining after failures."
                
                # Generate the frame with the selected font
                print(f"Creating frame with font={os.path.basename(font_path)}, background_style='{background_style}'")
                frame = create_text_image_frame(
                    width, height, text_lines, highlight_line_index, highlighted_text,
                    font_path, font_size, text_color, background_color, highlight_color,
                    blur_type, blur_radius, radial_sharp_radius_factor, vertical_spread_factor,
                    background_style  # Pass the background style parameter
                )
                
                frames.append(frame)
                frame_generated = True
                break

            except (FontLoadError, FontDrawError) as e:
                # Add to failed fonts list
                if font_try < MAX_FONT_RETRIES_PER_FRAME - 1: # Don't log exhaustively
                    print(f"  Font error on frame {frame_index}, attempt {font_try+1}: {e}")
                failed_fonts.add(font_path)
                # Remove from snippet's fonts if needed
                if font_path in unique_fonts_per_snippet[snippet_idx]:
                    unique_fonts_per_snippet[snippet_idx].remove(font_path)
                continue
            
            except Exception as e:
                print(f"Unexpected error on frame {frame_index}: {e}")
                return None, f"Unexpected error during frame generation: {e}"
        
        if not frame_generated:
            return None, f"Failed to generate frame {frame_index} after {MAX_FONT_RETRIES_PER_FRAME} font attempts."
        
        # Progress indicator for long-running generations
        if frame_index % max(1, num_frames // 10) == 0:
            print(f"  Generated frame {frame_index+1}/{num_frames}")

    # Create a video file
    output_filename = f"text_match_cut_{int(time.time())}.mp4"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    
    try:
        # Ensure frames is a list of numpy arrays
        if not isinstance(frames[0], np.ndarray):
             frames = [np.array(f) for f in frames]

        clip = ImageSequenceClip(frames, fps=fps)

        # Write video file using recommended settings
        # logger='bar' might not work well in web server logs, use None or default
        # Specify audio=False if there's no audio track
        # threads can speed up encoding, preset affects quality/speed balance
        clip.write_videofile(output_path,
                             codec='libx264', # Good compatibility
                             preset='medium', # Balance speed/quality (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
                             fps=fps,
                             threads=max(1, os.cpu_count() // 2), # Use half CPU cores
                             logger=None, # Avoid progress bar in server logs
                             audio=False) # Explicitly no audio

        clip.close() # Release resources
        print(f"\nVideo saved successfully as '{output_filename}'")

        # Optionally list failed fonts
        if failed_fonts:
            print("\nFonts that caused errors during generation:")
            for ff in sorted(list(failed_fonts)):
                print(f" - {os.path.basename(ff)}")

        return output_filename, None # Return filename on success, no error

    except Exception as e:
        print(f"\nError during video writing: {e}")
        traceback.print_exc()
        error_message = f"Error during video writing: {e}. Check server logs and FFmpeg installation/codec support (libx264)."
        # Clean up potentially partially written file
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass # Ignore cleanup error
        return None, error_message


# --- Video Generation Tracking ---
class VideoTracker:
    def __init__(self):
        self.generations = {}  # IP -> list of (timestamp, filename)
        self.video_status = {}  # video_id -> status info
    
    def can_generate(self, ip):
        today = datetime.now().date()
        count = sum(1 for ts, _ in self.generations.get(ip, [])
                   if ts.date() == today)
        return count < app.config['MAX_VIDEOS_PER_DAY']
    
    def add_generation(self, ip, video_id):
        if ip not in self.generations:
            self.generations[ip] = []
        self.generations[ip].append((datetime.now(), video_id))
        self.video_status[video_id] = {
            'status': 'processing',
            'created_at': datetime.now().isoformat(),
            'video_url': None
        }
        self.cleanup_old_entries()
    
    def get_status(self, video_id):
        """Get the current status of a video generation."""
        return self.video_status.get(video_id)
    
    def update_status(self, video_id, status, video_url=None):
        """Update the status of a video generation."""
        if video_id in self.video_status:
            self.video_status[video_id].update({
                'status': status,
                'video_url': video_url
            })
    
    def cleanup_old_entries(self):
        cutoff = datetime.now() - timedelta(minutes=app.config['CLEANUP_MINUTES'])
        for ip in list(self.generations.keys()):
            self.generations[ip] = [(ts, fn) for ts, fn in self.generations[ip]
                                  if ts > cutoff]
            if not self.generations[ip]:
                del self.generations[ip]

tracker = VideoTracker()

# --- Cleanup Function ---
def cleanup_old_videos():
    """Remove videos older than CLEANUP_MINUTES."""
    try:
        cutoff = datetime.now() - timedelta(minutes=app.config['CLEANUP_MINUTES'])
        output_dir = Path(app.config['UPLOAD_FOLDER'])
        for video_file in output_dir.glob('*.mp4'):
            if datetime.fromtimestamp(video_file.stat().st_mtime) < cutoff:
                video_file.unlink()
                print(f"Cleaned up old video: {video_file}")
    except Exception as e:
        print(f"Error during cleanup: {e}")

# --- Error Handlers ---
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify(error="Rate limit exceeded. Please try again later."), 429

@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify(error="Request too large."), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify(error="Internal server error. Please try again later."), 500

# --- Routes ---
@app.route('/', methods=['GET'])
def index():
    """Renders the main form page."""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
@limiter.limit("10 per minute")
def generate():
    """Handles form submission, triggers video generation."""
    try:
        # Check rate limits
        ip = get_remote_address()
        if not tracker.can_generate(ip):
            return jsonify(error="Daily video generation limit reached. Please try again tomorrow."), 429

        # Validate input parameters
        params = {
            'width': request.form.get('width', default=1024, type=int),
            'height': request.form.get('height', default=1024, type=int),
            'fps': request.form.get('fps', default=10, type=int),
            'duration': request.form.get('duration', default=5, type=int),
            'highlighted_text': request.form.get('highlighted_text', default="Missing Text"),
            'highlight_color': request.form.get('highlight_color', default='#FFFF00'),
            'text_color': request.form.get('text_color', default='#000000'),
            'background_color': request.form.get('background_color', default='#FFFFFF'),
            'blur_type': request.form.get('blur_type', default='gaussian'),
            'blur_radius': request.form.get('blur_radius', default=4.0, type=float),
            'background_style': request.form.get('background_style', default='solid'),
            'ai_enabled': True,
        }

        # Input validation
        if not params['highlighted_text']:
            flash('Highlighted text cannot be empty.', 'error')
            return redirect(url_for('index'))
        if not (1 <= params['fps'] <= 60):
            flash('FPS must be between 1 and 60.', 'error')
            return redirect(url_for('index'))
        if not (1 <= params['duration'] <= 60): # Limit duration
             flash('Duration must be between 1 and 60 seconds.', 'error')
             return redirect(url_for('index'))
        if not (256 <= params['width'] <= 4096) or not (256 <= params['height'] <= 4096):
             flash('Width and Height must be between 256 and 4096 pixels.', 'error')
             return redirect(url_for('index'))
        
        # Validate background style
        valid_styles = ['solid', 'newspaper', 'old_paper']
        if params['background_style'] not in valid_styles:
            params['background_style'] = 'solid'  # Default to solid if invalid

        # --- Generate a unique video ID ---
        video_id = str(uuid.uuid4())

        # Store in tracker and Redis
        tracker.add_generation(ip, video_id)
        set_video_status_redis(video_id, "processing", params=params)

        # Start background processing
        threading.Thread(
            target=generate_video_background,
            args=(video_id, params)
        ).start()

        # Redirect to the status page instead of rendering a template
        return redirect(url_for('status_page', video_id=video_id))

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        flash(error_msg, 'error')
        traceback.print_exc()
        return redirect(url_for('index'))


@app.route('/output/<filename>')
def download_file(filename):
    """Serves the generated video file for preview or download."""
    try:
        # Check if it's a preview request (no download parameter)
        as_attachment = request.args.get('download', 'false').lower() == 'true'
        
        # Set proper headers for video files
        response = send_from_directory(
            app.config["UPLOAD_FOLDER"], 
            filename,
            mimetype='video/mp4',
            as_attachment=as_attachment
        )
        
        # Add content disposition header for downloads
        if as_attachment:
            response.headers["Content-Disposition"] = f"attachment; filename={filename}"
        
        return response
        
    except FileNotFoundError:
        flash('Error: File not found. It might have been deleted or generation failed.', 'error')
        return redirect(url_for('index'))
    except Exception as e:
        print(f"Error serving file {filename}: {e}")
        flash('An error occurred while trying to serve the file.', 'error')
        return redirect(url_for('index'))

def upload_to_catbox(file_path):
    """Uploads a file to Catbox and returns the URL."""
    try:
        print(f"Attempting to upload file to Catbox: {file_path}")
        with open(file_path, 'rb') as f:
            files = {'fileToUpload': f}
            data = {'reqtype': 'fileupload'}
            print("Sending request to Catbox API...")
            response = requests.post(CATBOX_API_URL, files=files, data=data)
            response.raise_for_status()
            catbox_url = response.text.strip()
            # Clean any potential bad characters from the URL
            if catbox_url and isinstance(catbox_url, str):
                catbox_url = catbox_url.rstrip(';,')
            print(f"Successfully uploaded to Catbox. URL: {catbox_url}")
            return catbox_url
    except Exception as e:
        print(f"Error uploading to Catbox: {e}")
        traceback.print_exc()  # Print full error traceback
        return None

def store_video_metadata(video_id, catbox_url):
    """Stores video metadata in Redis."""
    if redis_client is None:
        print("Redis is not available, skipping metadata storage")
        return False
    try:
        # Clean the URL to remove any potential semicolons
        if isinstance(catbox_url, str):
            catbox_url = catbox_url.rstrip(';,')
            
        # Store with expiration of 30 days
        key = f"video:{video_id}"
        result = redis_client.setex(key, 30 * 24 * 60 * 60, catbox_url)
        print(f"Stored URL in Redis: key={key}, value={catbox_url}, result={result}")
        
        # Also update the status
        set_video_status_redis(video_id, "completed", video_url=catbox_url)
        return True
    except Exception as e:
        print(f"Error storing video metadata in Redis: {e}")
        return False

def get_video_url(video_id):
    """Retrieves video URL from Redis."""
    if redis_client is None:
        print("Redis is not available, cannot retrieve video URL")
        return None
    try:
        key = f"video:{video_id}"
        print(f"Checking Redis for URL key: {key}")
        url = redis_client.get(key)
        if url:
            print(f"Found URL in Redis for {video_id}: {url}")
            return url
        print(f"No URL found in Redis for ID: {video_id}")
        return None
    except Exception as e:
        print(f"Error retrieving video URL from Redis: {e}")
        return None

def generate_video_background(video_id, params):
    """Background task for video generation with Catbox upload."""
    try:
        print(f"Starting background video generation for {video_id}")
        set_video_status_redis(video_id, "processing")
        
        generated_filename, error = generate_video(params)
        
        if error:
            print(f"Video generation failed for {video_id}: {error}")
            set_video_status_redis(video_id, "failed", error=error)
            return
        
        # If successful, update status and upload to Catbox
        print(f"Video generation successful for {video_id}: {generated_filename}")
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], generated_filename)
        
        # Upload to Catbox for sharing
        catbox_url = upload_to_catbox(video_path)
        
        if catbox_url:
            print(f"Catbox upload successful for {video_id}: {catbox_url}")
            # Store in Redis and update status
            if redis_client is not None:
                store_video_metadata(video_id, catbox_url)
            set_video_status_redis(video_id, "completed", video_url=catbox_url)
        else:
            # If Catbox fails, still provide local URL
            print(f"Catbox upload failed for {video_id}, using local URL")
            local_url = url_for('download_file', filename=generated_filename, _external=True)
            set_video_status_redis(video_id, "completed", video_url=local_url)
    except Exception as e:
        print(f"Unexpected error during video generation for {video_id}: {e}")
        traceback.print_exc()
        set_video_status_redis(video_id, "failed", error=str(e))

@app.route('/api/generate', methods=['POST'])
@limiter.limit("10 per minute")
def api_generate():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Validate required fields
        required_fields = ['highlighted_text', 'width', 'height', 'duration']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Add default values for optional parameters (matching index.html defaults)
        params = {
            # Required parameters
            'width': data['width'],
            'height': data['height'],
            'duration': data['duration'],
            'highlighted_text': data['highlighted_text'],
            
            # Optional parameters with defaults from index.html
            'fps': data.get('fps', 5),                    # Default fps to 5
            'highlight_color': data.get('highlight_color', '#00f7ff'),  # Default from index.html
            'text_color': data.get('text_color', '#ffffff'),          # Default from index.html
            'background_color': data.get('background_color', '#0a0a0a'), # Default from index.html
            'blur_type': data.get('blur_type', 'radial'),             # Default from index.html
            'blur_radius': data.get('blur_radius', 4.0),              # Default from index.html
            'background_style': data.get('background_style', 'solid'),  # Default to solid
            'ai_enabled': data.get('ai_enabled', True)
        }

        # Generate unique video ID
        video_id = str(uuid.uuid4())
        # Set status in Redis as soon as video_id is created
        set_video_status_redis(video_id, "processing", params=params)
        
        # Check rate limits
        if not tracker.can_generate(request.remote_addr):
            return jsonify({
                'error': 'Rate limit exceeded',
                'message': 'Too many video generations. Please try again later.'
            }), 429

        # Validate specific parameters
        if not (1 <= params['fps'] <= 60):
            return jsonify({'error': 'FPS must be between 1 and 60'}), 400
        if not (1 <= params['duration'] <= 60):
            return jsonify({'error': 'Duration must be between 1 and 60 seconds'}), 400
        if not (256 <= params['width'] <= 4096) or not (256 <= params['height'] <= 4096):
            return jsonify({'error': 'Width and Height must be between 256 and 4096 pixels'}), 400
            
        # Validate background style
        valid_styles = ['solid', 'newspaper', 'old_paper']
        if params['background_style'] not in valid_styles:
            params['background_style'] = 'solid'  # Default to solid if invalid

        # Start processing in background
        tracker.add_generation(request.remote_addr, video_id)
        
        # Start the video generation in a background thread
        threading.Thread(
            target=generate_video_background,
            args=(video_id, params)
        ).start()
        
        # Return immediately with the video_id
        return jsonify({
            'status': 'processing',
            'video_id': video_id,
            'message': 'Video generation started. Poll /api/status/{video_id} for updates.',
            'status_url': url_for('api_status', video_id=video_id, _external=True)
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'error': 'Server error',
            'message': str(e)
        }), 500

@app.route('/api/status/<video_id>', methods=['GET'])
def api_status(video_id):
    """Returns the status of a video generation."""
    try:
        status = get_video_status_redis(video_id)
        
        if status:
            # Return the status
            return jsonify(status)
        
        # If status not found, check the video URL directly
        video_url = get_video_url(video_id)
        if video_url:
            # If URL exists but status doesn't, create a completed status
            try:
                if isinstance(video_url, bytes):
                    url_str = video_url.decode('utf-8').rstrip(';,')
                else:
                    url_str = str(video_url).rstrip(';,')
                
                # Create status data and store it for future requests
                status_data = {
                    "status": "completed",
                    "created_at": datetime.now().isoformat(),
                    "video_url": url_str,
                    "error": None
                }
                # Store the status for future requests
                set_video_status_redis(video_id, "completed", video_url=url_str)
                return jsonify(status_data)
            except Exception as e:
                print(f"Error processing video URL in status endpoint: {e}")
        
        # If nothing found, return not found
        return jsonify({
            'status': 'not_found',
            'error': 'Video ID not found'
        }), 404
    except Exception as e:
        print(f"Error in api_status endpoint: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'error': f'Server error: {e}'
        }), 500

@app.route('/api/video/<video_id>', methods=['GET'])
def api_video(video_id):
    """Get video URL for a generated video by ID."""
    try:
        print(f"API video request for ID: {video_id}")
        
        # DIRECT check of both Redis keys first
        if redis_client:
            # Try status key first (new format)
            status_key = f"status:{video_id}"
            status_data = redis_client.get(status_key)
            if status_data:
                try:
                    # Clean and parse the data
                    if isinstance(status_data, bytes):
                        status_str = status_data.decode('utf-8').rstrip(';,')
                    else:
                        status_str = str(status_data).rstrip(';,')
                    
                    # Try to repair common JSON errors
                    if '";' in status_str:
                        status_str = status_str.replace('";', '"')
                    
                    status_json = json.loads(status_str)
                    print(f"Status from Redis: {status_json}")
                    
                    if status_json.get('status') == 'completed' and status_json.get('video_url'):
                        # Found completed video
                        video_url = status_json.get('video_url')
                        if video_url:
                            # Ensure URL is clean
                            video_url = video_url.rstrip(';,') if isinstance(video_url, str) else video_url
                            return jsonify({
                                'status': 'success',
                                'video_id': video_id,
                                'video_url': video_url
                            })
                except Exception as e:
                    print(f"Error parsing status data: {e}")
            
            # Try direct URL key (legacy format)
            url_key = f"video:{video_id}"
            url_data = redis_client.get(url_key)
            if url_data:
                try:
                    if isinstance(url_data, bytes):
                        url_str = url_data.decode('utf-8').rstrip(';,')
                    else:
                        url_str = str(url_data).rstrip(';,')
                        
                    print(f"URL from Redis: {url_str}")
                    # Set status for future requests
                    set_video_status_redis(video_id, "completed", video_url=url_str)
                    return jsonify({
                        'status': 'success',
                        'video_id': video_id,
                        'video_url': url_str
                    })
                except Exception as e:
                    print(f"Error processing URL data: {e}")
        
        # If direct check failed, try standard helper methods as fallback
        status = get_video_status_redis(video_id)
        
        if status and status.get('status') == 'completed' and status.get('video_url'):
            # Ensure the URL is clean
            video_url = status.get('video_url').rstrip(';,') if isinstance(status.get('video_url'), str) else status.get('video_url')
            return jsonify({
                'status': 'success',
                'video_id': video_id,
                'video_url': video_url
            })
        elif status and status.get('status') == 'failed':
            return jsonify({
                'status': 'error',
                'error': status.get('error', 'Video generation failed')
            }), 400
        elif status and status.get('status') == 'processing':
            return jsonify({
                'status': 'processing',
                'message': 'Video is still being generated'
            })
        
        # If status not found or incomplete, try the legacy method (direct URL lookup)
        video_url = get_video_url(video_id)
        if video_url:
            try:
                if isinstance(video_url, bytes):
                    url_str = video_url.decode('utf-8').rstrip(';,')
                else:
                    url_str = str(video_url).rstrip(';,')
                
                # Also update the status in Redis for future requests
                set_video_status_redis(video_id, "completed", video_url=url_str)
                return jsonify({
                    'status': 'success',
                    'video_id': video_id,
                    'video_url': url_str
                })
            except Exception as e:
                print(f"Error converting URL from Redis: {e}")
        
        # If all else fails, return error
        return jsonify({
            'status': 'error',
            'error': 'Video not ready or not found'
        }), 404
    except Exception as e:
        print(f"Unexpected error in api_video endpoint: {e}")
        traceback.print_exc()  # Print the full traceback for debugging
        return jsonify({
            'status': 'error',
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/api/docs')
def api_docs():
    """Renders the API documentation page."""
    return render_template('api_docs.html')

@app.before_request
def before_request_cleanup():
    cleanup_old_videos()

# --- Redis Status Helpers ---
def set_video_status_redis(video_id, status, video_url=None, error=None, params=None):
    """Store video status information in Redis."""
    if redis_client is None:
        print(f"Cannot set video status: Redis client is None")
        return False
    
    key = f"status:{video_id}"
    
    # Clean the video_url if it exists to ensure no malformed characters
    if video_url and isinstance(video_url, str):
        video_url = video_url.strip().rstrip(';,')
        
    data = {
        "status": status,
        "created_at": datetime.now().isoformat(),
        "video_url": video_url,
        "error": error
    }
    
    # Store params if provided
    if params:
        data["params"] = params
    
    try:
        # Ensure clean JSON data - no trailing characters
        json_data = json.dumps(data)
        # Double check the JSON is valid (won't actually re-serialize)
        json.loads(json_data)
        
        # Store in Redis
        result = redis_client.setex(key, 30 * 24 * 60 * 60, json_data)
        print(f"Redis setex result for {key}: {result}")
        print(f"Status data stored in Redis: {json_data}")
        return True
    except Exception as e:
        print(f"Error setting video status in Redis for ID {video_id}: {e}")
        return False

def get_video_status_redis(video_id):
    """Retrieve video status information from Redis."""
    if redis_client is None:
        print(f"Cannot get video status: Redis client is None")
        return None
    
    key = f"status:{video_id}"
    try:
        print(f"Checking Redis for status key: {key}")
        data = redis_client.get(key)
        
        if data:
            try:
                # Ensure the data is properly formatted JSON
                if isinstance(data, bytes):
                    data_str = data.decode('utf-8').rstrip(';,')
                else:
                    data_str = str(data).rstrip(';,')
                
                status_data = json.loads(data_str)
                print(f"Found status data in Redis for {video_id}: {status_data}")
                return status_data
            except json.JSONDecodeError as je:
                print(f"Error decoding JSON from Redis for ID {video_id}: {je}")
                print(f"Malformed data: {data}")
                # Try to clean it up and retry
                try:
                    # Try to manually clean the data
                    if isinstance(data, bytes):
                        data_str = data.decode('utf-8')
                    else:
                        data_str = str(data)
                        
                    # Remove trailing characters that shouldn't be there
                    if '"video_url": "' in data_str and '";' in data_str:
                        data_str = data_str.replace('";', '"')
                    
                    status_data = json.loads(data_str)
                    print(f"Successfully cleaned and parsed data: {status_data}")
                    # Store corrected data back to Redis
                    set_video_status_redis(
                        video_id, 
                        status_data.get("status", "completed"), 
                        video_url=status_data.get("video_url"),
                        error=status_data.get("error")
                    )
                    return status_data
                except Exception:
                    print(f"Failed to clean malformed data")
                    pass
                    
        else:
            print(f"No status data found in Redis for {video_id}")
            # Check if we have a URL stored in the legacy format
            legacy_url = get_video_url(video_id)
            if legacy_url:
                print(f"Found legacy URL for {video_id}: {legacy_url}")
                # Create status data from legacy URL
                try:
                    url_str = legacy_url.decode('utf-8').rstrip(';,')
                    status_data = {
                        "status": "completed",
                        "created_at": datetime.now().isoformat(),
                        "video_url": url_str,
                        "error": None
                    }
                    # Store it in the new format
                    set_video_status_redis(video_id, "completed", video_url=url_str)
                    return status_data
                except Exception as e:
                    print(f"Error processing legacy URL: {e}")
        return None
    except Exception as e:
        print(f"Error getting video status from Redis for ID {video_id}: {e}")
        return None

@app.route('/status/<video_id>', methods=['GET'])
def status_page(video_id):
    """Renders the status page for a video generation."""
    # Get video text from Redis if available
    text = "your text"
    
    try:
        status = get_video_status_redis(video_id)
        if status and 'params' in status and 'highlighted_text' in status['params']:
            text = status['params']['highlighted_text']
    except Exception as e:
        print(f"Could not retrieve highlighted text for {video_id}: {e}")
    
    return render_template('status.html', 
                          video_id=video_id, 
                          highlighted_text=text)

# --- Main Execution ---
if __name__ == '__main__':
    # Use host='0.0.0.0' to make accessible on your network (use with caution)
    # debug=True automatically reloads on code changes, but disable for production
    app.run(debug=True, host='127.0.0.1', port=5000)