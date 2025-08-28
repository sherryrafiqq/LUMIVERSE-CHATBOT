# InsideOut Chatbot API

An empathetic AI chatbot built with FastAPI that provides emotional support and conversation. The chatbot uses Cohere's AI models to detect emotions and generate appropriate responses.

## 🚀 Features

- **Emotion Detection**: Analyzes user messages to detect 6 emotions (joy, sadness, anger, fear, disgust, neutral)
- **Empathetic Responses**: Generates warm, supportive responses based on detected emotions
- **Production Ready**: Built with FastAPI, includes error handling, rate limiting, and health checks
- **Cross-Platform**: Works on Railway, Render, PythonAnywhere, and local development
- **Modern UI**: Beautiful HTML test interface included

## 📋 Prerequisites

- Python 3.11+
- Cohere API key (get one at [cohere.ai](https://cohere.ai))

## 🛠️ Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd IoT2
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Copy the template
   # No need to copy - use variables.env directly
   
   # Edit variables.env and add your API keys
   COHERE_API_KEY=your_actual_api_key_here
   SUPABASE_URL=https://qvqbhoptpecvflidiqik.supabase.co
   SUPABASE_ANON_KEY=your_supabase_anon_key_here
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

5. **Test the API**
   - Open `http://localhost:8000/test` in your browser
   - Or visit `http://localhost:8000/docs` for API documentation

## 🚂 Railway Deployment

### Quick Deploy (Recommended)

1. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "Ready for Railway deployment"
   git push origin main
   ```

2. **Deploy on Railway**
   - Go to [railway.app](https://railway.app)
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Railway will automatically detect it's a Python app

3. **Set Environment Variables**
   In the Railway dashboard, go to your project → Variables tab and add:
   ```
   COHERE_API_KEY=your_actual_cohere_api_key
   SUPABASE_URL=https://qvqbhoptpecvflidiqik.supabase.co
   SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InF2cWJob3B0cGVjdmZsaWRpcWlrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQ1OTU4MTUsImV4cCI6MjA3MDE3MTgxNX0.RMohPWr8VmsC5lvFmN6Ys27CWByySwdlsks09O9N97k
   RAILWAY=true
   ```

4. **Deploy!** Railway will automatically build and deploy your app.

### Alternative: Railway CLI

1. **Install Railway CLI**
   ```bash
   npm install -g @railway/cli
   ```

2. **Login and deploy**
   ```bash
   railway login
   railway init
   railway up
   ```

3. **Set environment variables**
   ```bash
   railway variables set COHERE_API_KEY=your_actual_api_key_here
   railway variables set SUPABASE_URL=https://qvqbhoptpecvflidiqik.supabase.co
   railway variables set SUPABASE_ANON_KEY=your_supabase_anon_key_here
   railway variables set RAILWAY=true
   ```

## 🔧 Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `COHERE_API_KEY` | Yes | Your Cohere API key | None |
| `SUPABASE_URL` | Yes | Your Supabase project URL | None |
| `SUPABASE_ANON_KEY` | Yes | Your Supabase anonymous key | None |
| `DEBUG` | No | Enable debug mode | `False` |
| `RAILWAY` | No | Set to `true` on Railway | Auto-detected |

## 📡 API Endpoints

### Health Check
- `GET /` - Root endpoint with basic info
- `GET /health` - Detailed health check
- `GET /health/detailed` - Comprehensive health status

### Chat
- `GET /start` - Start a new conversation
- `POST /chat` - Send a message and get response

### Testing
- `GET /test` - HTML test interface
- `POST /test-chat` - Test chat endpoint

## 🎯 API Usage

### Send a message
```bash
curl -X POST "https://your-app.railway.app/api" \
  -H "Content-Type: application/json" \
  -d '{
    "action": "chat",
    "user_id": "d32d5387-94a0-4ce2-ad40-12e205bd25ae",
    "message": "I feel stressed"
  }'
```

### Response format
```json
{
  "success": true,
  "message": "Chat response generated",
  "data": {
    "emotion": "sadness",
    "reply": "I'm so sorry you're feeling sad today. 💙 It's completely okay to feel this way, and I'm here to listen..."
  },
  "timestamp": "2024-01-01T12:00:00.000000"
}
```

### Request format

```json
{
  "action": "chat",
  "user_id": "d32d5387-94a0-4ce2-ad40-12e205bd25ae",
  "message": "I feel stressed"
}
```

## 🎨 Test Interface

The application includes a beautiful HTML test interface accessible at `/test`. Features:

- Real-time chat interface
- Emotion detection display
- API status monitoring
- Responsive design
- Typing indicators

## 🔒 Security Features

- Rate limiting (60 requests per minute)
- Input validation and sanitization
- CORS protection
- Error handling without sensitive data leakage
- Request timeout protection

## 🐛 Troubleshooting

### Common Issues

1. **AI service unavailable**
   - Check if `COHERE_API_KEY` is set correctly
   - Verify your Cohere API key is valid and has credits

2. **Deployment fails**
   - Ensure all files are committed to your repository
   - Check that `requirements.txt` and `Procfile` are in the root directory
   - Verify Python version in `runtime.txt`

3. **CORS errors**
   - The app automatically configures CORS for different platforms
   - For custom domains, update the `allowed_origins` list in the code

### Health Check

Visit `/health` to check the status of all components:
- API status
- AI service availability
- Environment variables
- Overall system health

## 📝 File Structure

```
IoT2/
├── main.py             # Main FastAPI application
├── requirements.txt    # Python dependencies
├── Procfile           # Railway deployment config
├── runtime.txt        # Python version
├── variables.env      # Environment variables template
├── test_api.html      # Test interface
└── README.md          # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

If you encounter any issues:

1. Check the health endpoint: `/health`
2. Review the logs in Railway dashboard
3. Ensure all environment variables are set correctly
4. Verify your Cohere API key is valid

---

**Built with ❤️ using FastAPI and Cohere AI**
