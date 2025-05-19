# Text Match Cut API Documentation

Welcome to the Text Match Cut API documentation. This API allows you to programmatically generate text highlight videos with AI-generated content.

## Base URL

```
https://your-domain.com/api
```

## Authentication

Currently, the API uses IP-based rate limiting. No authentication is required.

## Rate Limits

- 200 requests per day per IP
- 50 requests per hour per IP
- 10 video generations per minute per IP
- 10 videos per day per IP

## Endpoints

### Generate Video

Creates a new video generation request.

```http
POST /api/generate
```

#### Request Body

```json
{
    "highlighted_text": "Your text here",
    "width": 1920,
    "height": 1080,
    "duration": 5,
    "fps": 5,
    "highlight_color": "#00f7ff",
    "text_color": "#ffffff",
    "background_color": "#0a0a0a",
    "background_style": "solid",
    "blur_type": "radial",
    "blur_radius": 4.0,
    "ai_enabled": true
}
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| highlighted_text | string | Yes | - | Text to highlight in the video |
| width | integer | Yes | - | Video width in pixels (256-4096) |
| height | integer | Yes | - | Video height in pixels (256-4096) |
| duration | integer | Yes | - | Video duration in seconds (1-60) |
| fps | integer | No | 5 | Frames per second (1-60) |
| highlight_color | string | No | "#00f7ff" | Color of the highlight box (hex) |
| text_color | string | No | "#ffffff" | Color of the text (hex) |
| background_color | string | No | "#0a0a0a" | Background color (hex) |
| background_style | string | No | "solid" | Background style ("solid", "newspaper", or "old_paper") |
| blur_type | string | No | "radial" | Type of blur ("gaussian" or "radial") |
| blur_radius | float | No | 4.0 | Blur effect radius |
| ai_enabled | boolean | No | true | Whether to use AI for text generation |

#### Response

```json
{
  "created_at": "2025-05-18T13:16:01.889647",
  "error": null,
  "status": "processing",
  "video_url": "https://files.catbox.moe/k5phm1.mp4"
}
    
```

### Check Status

Check the status of a video generation request.

```http
GET /api/status/{video_id}
```

#### Response

```json
{
    "status": "processing",
    "created_at": "2024-05-17T15:39:23.123456",
    "video_url": null,
    "error": null
}
```

Status values:
- `processing`: Video is being generated
- `completed`: Video is ready
- `failed`: Generation failed

### Get Video

Get the video URL once generation is complete.

```http
GET /api/video/{video_id}
```

#### Response

```json
{
    "status": "success",
    "video_id": "550e8400-e29b-41d4-a716-446655440000",
    "video_url": "https://files.catbox.moe/agsbdsa.mp4"
}
```

## Error Responses

### Rate Limit Exceeded

```json
{
    "error": "Rate limit exceeded",
    "message": "Too many video generations. Please try again later."
}
```

### Invalid Request

```json
{
    "error": "Missing required field: highlighted_text"
}
```

### Video Not Found

```json
{
    "status": "error",
    "error": "Video not ready or not found"
}
```

### Server Error

```json
{
    "status": "error",
    "error": "Server error: [error message]"
}
```

## Example Usage

### cURL

```bash
# Generate video
curl -X POST https://your-domain.com/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "highlighted_text": "Hello World",
    "width": 1920,
    "height": 1080,
    "duration": 5
  }'

# Check status
curl https://your-domain.com/api/status/550e8400-e29b-41d4-a716-446655440000

# Get video
curl https://your-domain.com/api/video/550e8400-e29b-41d4-a716-446655440000
```

### Python

```python
import requests

# Generate video
response = requests.post('https://your-domain.com/api/generate', json={
    'highlighted_text': 'Hello World',
    'width': 1920,
    'height': 1080,
    'duration': 5,
    'background_style': 'newspaper'  # Example with newspaper style
})
video_id = response.json()['video_id']

# Check status
status = requests.get(f'https://your-domain.com/api/status/{video_id}').json()

# Get video URL when ready
if status['status'] == 'completed':
    video = requests.get(f'https://your-domain.com/api/video/{video_id}').json()
    print(f"Video URL: {video['video_url']}")
```

## Notes

- Videos are automatically deleted after 1 day
- Maximum video duration is 60 seconds
- Maximum video dimensions are 4096x4096 pixels
- Minimum video dimensions are 256x256 pixels
- FPS must be between 1 and 60
- Duration must be between 1 and 60 seconds
- Video URLs are stored in Redis for 30 days
- The API uses Redis for status tracking and URL storage
- Failed video generations are automatically cleaned up

---

Made with ❤️ by R3AP3R editz 
