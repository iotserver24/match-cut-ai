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
    "highlight_color": "#ff0000",
    "text_color": "#ffffff",
    "background_color": "#000000",
    "blur_type": "gaussian",
    "blur_radius": 10
}
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| highlighted_text | string | Yes | - | Text to highlight in the video |
| width | integer | Yes | - | Video width in pixels (256-4096) |
| height | integer | Yes | - | Video height in pixels (256-4096) |
| duration | integer | Yes | - | Video duration in seconds (1-60) |
| highlight_color | string | No | "#FFFF00" | Color of the highlight box (hex) |
| text_color | string | No | "#000000" | Color of the text (hex) |
| background_color | string | No | "#FFFFFF" | Background color (hex) |
| blur_type | string | No | "gaussian" | Type of blur ("gaussian" or "radial") |
| blur_radius | float | No | 4.0 | Blur effect radius |

#### Response

```json
{
    "video_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "processing",
    "message": "Video generation started",
    "status_url": "/api/status/550e8400-e29b-41d4-a716-446655440000"
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
    "video_url": null
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
    "video_url": "https://your-domain.com/video/550e8400-e29b-41d4-a716-446655440000.mp4",
    "download_url": "https://your-domain.com/download/550e8400-e29b-41d4-a716-446655440000"
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
    "error": "Video ID not found"
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
    'duration': 5
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

---

Made with ❤️ by R3AP3R editz 