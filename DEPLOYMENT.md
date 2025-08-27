# 🚂 Railway Deployment Checklist

## ✅ Pre-Deployment Checklist

### Files Required
- [x] `main.py` - Main FastAPI application
- [x] `requirements.txt` - Python dependencies
- [x] `Procfile` - Railway deployment config
- [x] `runtime.txt` - Python version
- [x] `.gitignore` - Exclude unnecessary files

### Environment Variables Needed
- [ ] `COHERE_API_KEY` - Your Cohere API key
- [ ] `SUPABASE_URL` - https://qvqbhoptpecvflidiqik.supabase.co
- [ ] `SUPABASE_ANON_KEY` - Your Supabase anonymous key
- [ ] `RAILWAY` - Set to `true`

## 🚀 Deployment Steps

### 1. Prepare Your Code
```bash
# Check if everything is ready
python railway_setup.py

# Commit your changes
git add .
git commit -m "Ready for Railway deployment"
git push origin main
```

### 2. Deploy on Railway
1. Go to [railway.app](https://railway.app)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your repository
5. Wait for Railway to detect Python app

### 3. Set Environment Variables
In Railway dashboard → Variables tab:
```
COHERE_API_KEY=your_actual_cohere_api_key
SUPABASE_URL=https://qvqbhoptpecvflidiqik.supabase.co
SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InF2cWJob3B0cGVjdmZsaWRpcWlrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQ1OTU4MTUsImV4cCI6MjA3MDE3MTgxNX0.RMohPWr8VmsC5lvFmN6Ys27CWByySwdlsks09O9N97k
RAILWAY=true
```

### 4. Test Your Deployment
- Visit your Railway URL
- Test the health endpoint: `https://your-app.railway.app/health`
- Test the chat endpoint: `https://your-app.railway.app/test`

## 🔧 Troubleshooting

### Common Issues
1. **Build fails** - Check requirements.txt and runtime.txt
2. **Environment variables missing** - Verify all variables are set in Railway dashboard
3. **App crashes** - Check Railway logs for errors
4. **Supabase connection fails** - Verify Supabase credentials

### Useful Commands
```bash
# Check Railway status
railway status

# View logs
railway logs

# Open Railway dashboard
railway open
```

## 📱 FlutterFlow Integration

Once deployed, your FlutterFlow app can use:
- **API URL**: `https://your-app.railway.app`
- **Chat Endpoint**: `POST /chat`
- **Registration**: Send `"register:username"` as first message

## 🎉 Success!

Your InsideOut chatbot is now live on Railway! 🚀
