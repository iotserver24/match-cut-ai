# Text Match Cut Video Generator

A powerful web application that creates stunning text highlight videos with AI-generated content. Built with Flask and modern web technologies.

![Made by: R3AP3R editz](https://img.shields.io/badge/Made%20by-R3AP3R%20editz-blue)

## 🌟 Features

- **AI-Powered Text Generation**: Uses Pollinations API for intelligent text generation
- **Beautiful Text Effects**: Multiple blur effects (Gaussian and Radial)
- **Customizable Design**: 
  - Adjustable video dimensions
  - Custom colors for text and highlights
  - Configurable blur effects
  - Multiple font support
- **Modern UI**: Elegant, futuristic interface with real-time preview
- **REST API**: Full API support for integration with other applications
- **Rate Limiting**: Built-in protection against abuse
- **Automatic Cleanup**: Removes old videos to save space

## 🚀 Quick Start

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/text-match-cut.git
cd text-match-cut
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
python app.py
```

4. **Access the application**:
Open your browser and visit `http://localhost:5000`

## 🛠️ Configuration

The application can be configured through environment variables:

- `PORT`: Server port (default: 5000)
- `MAX_VIDEOS_PER_DAY`: Daily video limit per IP (default: 10)
- `CLEANUP_DAYS`: Days to keep videos (default: 1)

## 📝 Usage

### Web Interface

1. Enter your highlighted text
2. Adjust video settings:
   - Width and height
   - Duration
   - Colors
   - Blur effects
3. Click "Generate Video"
4. Preview and download your video

### API Usage

See [API Documentation](API.md) for detailed information about the REST API.

## 🔧 Technical Details

### Dependencies

- Flask: Web framework
- MoviePy: Video generation
- Pillow: Image processing
- Requests: API calls
- Flask-Limiter: Rate limiting

### Directory Structure

```
text-match-cut/
├── app.py              # Main application
├── requirements.txt    # Dependencies
├── templates/          # HTML templates
│   └── index.html     # Main interface
├── static/            # Static files
├── output/            # Generated videos
└── fonts/            # Custom fonts
```

## 🔒 Security

- Rate limiting on all endpoints
- Input validation
- Secure file handling
- Automatic cleanup of old files

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Pollinations API for text generation
- All contributors and users

## 📞 Support

For support, please open an issue in the GitHub repository.

---

Made with ❤️ by R3AP3R editz
