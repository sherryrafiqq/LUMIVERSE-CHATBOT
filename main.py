# main.py - Production-ready FastAPI for Railway (Chatbot Only)
import os
import sys
import time
from datetime import datetime, date
from typing import Optional, List, Dict, Any
import logging

# Configure logging for Railway.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/insideout_api.log') if os.path.exists('/tmp') else logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add current directory to Python path for Railway
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from fastapi import FastAPI, HTTPException, status, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, validator
    from langchain_core.prompts import PromptTemplate
    from langchain_cohere import ChatCohere
    import traceback
    from dotenv import load_dotenv
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)

# Load environment variables
load_dotenv("variables.env")

# -------------------------
# 🔧 Configuration
# -------------------------

# Environment Variables - No hardcoded defaults for security
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Platform specific configurations
IS_PYTHONANYWHERE = os.getenv("PYTHONANYWHERE_SITE", "").startswith("www.pythonanywhere.com")
IS_RENDER = os.getenv("RENDER", "False").lower() == "true"
IS_RAILWAY = os.getenv("RAILWAY", "False").lower() == "true" or os.getenv("RAILWAY_ENVIRONMENT", "").lower() == "production"
DEBUG_MODE = os.getenv("DEBUG", "False").lower() == "true"

# Force production mode on cloud platforms
if IS_PYTHONANYWHERE or IS_RENDER or IS_RAILWAY:
    DEBUG_MODE = False

# Set Cohere API key in environment
if COHERE_API_KEY:
    os.environ["COHERE_API_KEY"] = COHERE_API_KEY

# FastAPI App Configuration
app = FastAPI(
    title="InsideOut Chatbot API",
    version="1.0.0",
    description="An empathetic AI chatbot for emotion support",
    docs_url="/docs" if DEBUG_MODE else None,  # Disable docs in production
    redoc_url="/redoc" if DEBUG_MODE else None,
    debug=DEBUG_MODE
)

# -------------------------
# 🌐 CORS Configuration for Flutter
# -------------------------
# More restrictive CORS for production
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:8080",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:3000",
    "file://",  # Allow local file access
    "null",     # Allow local file access
    "https://your-flutter-app-domain.com",  # Replace with your actual domain
]

# For local development and cloud platforms, allow all origins
if not IS_PYTHONANYWHERE or IS_RENDER or IS_RAILWAY:
    allowed_origins = ["*"]

# For PythonAnywhere, use more restrictive CORS
if IS_PYTHONANYWHERE:
    allowed_origins = [
        "https://sherry38.pythonanywhere.com",
        "https://www.sherry38.pythonanywhere.com",
        "http://localhost:3000",
        "http://localhost:8080",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:3000",
    ]

if DEBUG_MODE:
    allowed_origins.append("*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True if IS_PYTHONANYWHERE else False,  # Enable credentials for PythonAnywhere only
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# -------------------------
# 📁 Static Files (for PythonAnywhere)
# -------------------------
try:
    # Mount static files if they exist
    static_dir = os.path.join(current_dir, "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        logger.info("Static files mounted at /static")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

# -------------------------
# 📋 Pydantic Models (Flutter Compatible)
# -------------------------

class ApiResponse(BaseModel):
    """Standardized API response for Flutter"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str = datetime.utcnow().isoformat()

class ChatRequest(BaseModel):
    message: str
    
    @validator('message')
    def validate_message(cls, v):
        if len(v.strip()) == 0:
            raise ValueError('Message cannot be empty')
        if len(v) > 1000:
            raise ValueError('Message too long (max 1000 characters)')
        return v.strip()

# -------------------------
# 🤖 AI Configuration (Optimized for Speed)
# -------------------------

# Initialize AI models with error handling
AI_AVAILABLE = False
llm = None
emotion_chain = None
response_chain = None

def initialize_ai():
    """Initialize AI models with retry logic"""
    global AI_AVAILABLE, llm, emotion_chain, response_chain
    
    logger.info(f"Initializing AI... COHERE_API_KEY present: {bool(COHERE_API_KEY)}")
    
    if not COHERE_API_KEY:
        logger.warning("Cohere API key not provided - AI features disabled")
        return False
    
    try:
        # Optimized LLM configuration for faster response generation
        llm = ChatCohere(
            model="command-r-plus", 
            temperature=0.1, 
            max_tokens=150,
            p=0.9,  # Nucleus sampling for faster generation
            k=50    # Top-k sampling for faster generation
        )
        
        emotion_prompt = PromptTemplate.from_template("""
You are an expert emotion classifier. Analyze the message and classify it into exactly one of these 6 categories:joy, sadness, anger, fear, disgust, neutral. Choose the closest match from: joy, sadness, anger, fear, disgust.
If the message is completely neutral with no emotional tone, respond with: neutral
Message: {message}
Respond with ONLY one word: joy, sadness, anger, fear, disgust, or neutral
""")

        response_prompt = PromptTemplate.from_template("""
You are a warm, empathetic friend. Respond briefly (4-5 sentences max) but with genuine care.

User's message: {message}
Detected emotion: {emotion}

Guidelines:
- Use relevant emojis
- Acknowledge their emotion briefly and validate it
- Use warm, natural language
- If they're sad/angry/fearful: offer comfort and support
- If they're joyful: share their happiness
- If neutral: gently invite them to share feelings
- For suicidal/self-harm/extreme anxiety or panic attack thoughts: respond with immediate care, send them the link for Egypt's General Secretariat of Mental Health: "https://mentalhealth.mohp.gov.eg/" and Egypt's mental health holtines: (16328, 105 and 15335), and encourage professional help. Reassure user that it's gonna get better, that you're by their side and there are people available to provide help 24/7.

Keep responses concise but caring. Sound like a compassionate human friend.
""")

        emotion_chain = emotion_prompt | llm
        response_chain = response_prompt | llm
        AI_AVAILABLE = True
        logger.info("AI models initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize AI models: {e}")
        AI_AVAILABLE = False
        return False

# Initialize AI on startup
initialize_ai()

# -------------------------
# 🚀 Application Startup
# -------------------------

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting InsideOut API...")
    
    # Initialize AI
    if initialize_ai():
        logger.info("✅ AI services initialized")
    else:
        logger.warning("⚠️ AI services failed to initialize - chat features will be unavailable")
    
    # Check environment variables
    missing_vars = []
    if not COHERE_API_KEY:
        missing_vars.append("COHERE_API_KEY")
    
    if missing_vars:
        logger.warning(f"⚠️ Missing environment variables: {', '.join(missing_vars)}")
    else:
        logger.info("✅ All environment variables configured")
    
    logger.info("🚀 InsideOut API startup complete")

# -------------------------
# 🚨 Global Exception Handler & Crash Prevention
# -------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors with crash prevention"""
    logger.error(f"Unhandled exception: {exc}\nTraceback: {traceback.format_exc()}")
    
    # Prevent sensitive information leakage
    error_message = "An unexpected error occurred. Please try again later."
    
    # Log additional context for debugging
    logger.error(f"Request path: {request.url.path}")
    logger.error(f"Request method: {request.method}")
    logger.error(f"User agent: {request.headers.get('user-agent', 'Unknown')}")
    
    return JSONResponse(
        status_code=500,
        content=ApiResponse(
            success=False,
            message="Internal server error",
            error=error_message
        ).dict()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ApiResponse(
            success=False,
            message=str(exc.detail),
            error=str(exc.detail)
        ).dict()
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors"""
    logger.warning(f"Validation error: {exc}")
    
    return JSONResponse(
        status_code=400,
        content=ApiResponse(
            success=False,
            message="Validation error",
            error=str(exc)
        ).dict()
    )

@app.exception_handler(TimeoutError)
async def timeout_error_handler(request: Request, exc: TimeoutError):
    """Handle timeout errors"""
    logger.error(f"Timeout error: {exc}")
    
    return JSONResponse(
        status_code=408,
        content=ApiResponse(
            success=False,
            message="Request timeout",
            error="The request took too long to process. Please try again."
        ).dict()
    )

# -------------------------
# 🛡️ Request Rate Limiting & Security
# -------------------------

from collections import defaultdict

# Simple in-memory rate limiting (in production, use Redis)
request_counts = defaultdict(list)
MAX_REQUESTS_PER_MINUTE = 60

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware to prevent abuse"""
    client_ip = request.client.host if request.client else "unknown"
    current_time = time.time()
    
    # Clean old requests (older than 1 minute)
    request_counts[client_ip] = [req_time for req_time in request_counts[client_ip] 
                                if current_time - req_time < 60]
    
    # Check rate limit
    if len(request_counts[client_ip]) >= MAX_REQUESTS_PER_MINUTE:
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return JSONResponse(
            status_code=429,
            content=ApiResponse(
                success=False,
                message="Rate limit exceeded",
                error="Too many requests. Please try again later."
            ).dict()
        )
    
    # Add current request
    request_counts[client_ip].append(current_time)
    
    # Add request timing
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Log slow requests
    if process_time > 5.0:  # Log requests taking more than 5 seconds
        logger.warning(f"Slow request: {request.url.path} took {process_time:.2f}s")
    
    return response

# -------------------------
# 🔄 Health Check & Recovery
# -------------------------

@app.get("/health/detailed", response_model=ApiResponse)
async def detailed_health_check():
    """Detailed health check with component status"""
    health_status = {
        "api_status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {}
    }
    
    # Check AI service
    try:
        if AI_AVAILABLE and llm:
            # Simple test with AI service
            test_result = emotion_chain.invoke({"message": "test"})
            health_status["components"]["ai_service"] = "healthy"
        else:
            health_status["components"]["ai_service"] = "unavailable"
    except Exception as e:
        logger.error(f"AI service health check failed: {e}")
        health_status["components"]["ai_service"] = "unhealthy"
    
    # Check environment variables
    missing_vars = []
    if not COHERE_API_KEY:
        missing_vars.append("COHERE_API_KEY")
    
    health_status["components"]["environment"] = "healthy" if not missing_vars else f"missing: {', '.join(missing_vars)}"
    
    # Overall status
    all_healthy = all(status == "healthy" for status in health_status["components"].values())
    health_status["api_status"] = "healthy" if all_healthy else "degraded"
    
    return ApiResponse(
        success=all_healthy,
        message="Detailed health check completed",
        data=health_status
    )

# -------------------------
# 🔧 Graceful Shutdown
# -------------------------

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("Application shutting down...")
    
    # Clear rate limiting data
    request_counts.clear()
    logger.info("Rate limiting data cleared")
    
    logger.info("Application shutdown complete")

# -------------------------
# 🛣️ API Routes (Flutter Compatible)
# -------------------------

@app.get("/", response_model=ApiResponse)
async def root():
    """Root endpoint - health check"""
    return ApiResponse(
        success=True,
        message="InsideOut Chatbot API is running!",
        data={
            "version": "1.0.0",
            "status": "healthy",
            "ai_available": AI_AVAILABLE,
            "environment": "production" if (IS_PYTHONANYWHERE or IS_RENDER or IS_RAILWAY) else "development"
        }
    )

@app.get("/health", response_model=ApiResponse)
async def health_check():
    """Detailed health check for monitoring"""
    health_data = {
        "api_status": "healthy",
        "ai_available": AI_AVAILABLE,
        "timestamp": datetime.utcnow().isoformat(),
        "environment": "production" if (IS_PYTHONANYWHERE or IS_RENDER or IS_RAILWAY) else "development"
    }
    
    return ApiResponse(
        success=True,
        message="Health check completed",
        data=health_data
    )

@app.get("/start", response_model=ApiResponse)
async def start_conversation():
    """Start a new conversation"""
    return ApiResponse(
        success=True,
        message="Conversation started",
        data={
            "emotion": "neutral",
            "reply": "Hi there! I'm here for you. How are you feeling today?",
            "user": "anonymous"
        }
    )

def parse_emotion_response(raw_response: str) -> str:
    """Parse and validate emotion response from AI"""
    
    # Valid emotions (exactly what we want)
    valid_emotions = ['joy', 'sadness', 'anger', 'fear', 'disgust', 'neutral']
    
    # Clean the response
    cleaned = raw_response.strip().lower()
    
    # Direct match
    if cleaned in valid_emotions:
        return cleaned
    
    # Handle "no emotion" -> "neutral"  
    if 'no emotion' in cleaned or cleaned == 'no_emotion':
        return 'neutral'
    
    # Check if any valid emotion is contained in the response
    for emotion in valid_emotions:
        if emotion in cleaned:
            return emotion
    
    # If nothing matches, default to neutral
    logger.warning(f"Unrecognized emotion '{cleaned}', defaulting to neutral")
    return 'neutral'

@app.post("/chat", response_model=ApiResponse)
async def chat_endpoint(req: ChatRequest):
    """Main chat endpoint with crash prevention"""
    # Debug logging
    logger.info(f"Chat endpoint called. AI_AVAILABLE: {AI_AVAILABLE}, emotion_chain: {emotion_chain is not None}, response_chain: {response_chain is not None}")
    
    if not AI_AVAILABLE:
        logger.error("AI service not available - returning error response")
        return ApiResponse(
            success=False,
            message="Chat unavailable",
            error="AI service is not available",
            data={
                "emotion": "neutral",
                "reply": "I'm here to listen and support you. Could you tell me more about how you're feeling?"
            }
        )
    
    try:
        # Input validation and sanitization
        if not req.message or len(req.message.strip()) == 0:
            return ApiResponse(
                success=False,
                message="Invalid message",
                error="Message cannot be empty"
            )
        
        # Limit message length to prevent abuse
        if len(req.message) > 1000:
            return ApiResponse(
                success=False,
                message="Message too long",
                error="Message must be less than 1000 characters"
            )
        
        # Get emotion detection with timeout protection
        detected_emotion = 'neutral'
        try:
            import asyncio
            # Set timeout for AI calls (10 seconds)
            emotion_result = await asyncio.wait_for(
                asyncio.to_thread(emotion_chain.invoke, {"message": req.message}),
                timeout=10.0
            )
            
            # Extract and clean the raw response
            raw_emotion = emotion_result.content.strip()
            
            # Log the raw emotion detection for debugging
            logger.info(f"Raw emotion detection: '{raw_emotion}' for message: '{req.message[:50]}...'")
            
            # Parse the emotion from the response
            detected_emotion = parse_emotion_response(raw_emotion)
            
            logger.info(f"Final parsed emotion: '{detected_emotion}'")
            
            # Normalize emotion (this validation is now redundant but kept for safety)
            if detected_emotion not in ['joy', 'sadness', 'anger', 'fear', 'disgust', 'neutral']:
                logger.warning(f"Unknown emotion detected: '{detected_emotion}', defaulting to neutral")
                detected_emotion = 'neutral'
                
        except asyncio.TimeoutError:
            logger.warning(f"Emotion detection timeout")
            detected_emotion = 'neutral'
        except Exception as e:
            logger.error(f"Emotion detection failed: {e}")
            detected_emotion = 'neutral'
        
        # Generate response with timeout protection
        reply = "I'm here to listen and support you. Could you tell me more about how you're feeling?"
        try:
            import asyncio
            start_time = time.time()
            # Set timeout for AI calls (20 seconds) - optimized for faster responses
            bot_response = await asyncio.wait_for(
                asyncio.to_thread(response_chain.invoke, {
                    "emotion": detected_emotion,
                    "message": req.message
                }),
                timeout=20.0
            )
            response_time = time.time() - start_time
            logger.info(f"Response generated in {response_time:.2f} seconds")
            reply = bot_response.content.strip()
            
            # Validate response
            if not reply or len(reply.strip()) == 0:
                reply = "I'm here to listen and support you. Could you tell me more about how you're feeling?"
            
            # Limit response length
            if len(reply) > 500:
                reply = reply[:500] + "..."
                
        except asyncio.TimeoutError:
            logger.warning(f"Response generation timeout")
            reply = "I'm taking a moment to think about your message. Could you tell me more about how you're feeling?"
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            reply = "I'm here to listen and support you. Could you tell me more about how you're feeling?"
        
        # Ensure we always return the expected structure
        return ApiResponse(
            success=True,
            message="Chat response generated",
            data={
                "emotion": detected_emotion,
                "reply": reply
            }
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ApiResponse(
            success=False,
            message="Chat processing failed",
            error="Unable to process your message right now. Please try again in a moment.",
            data={
                "emotion": "neutral",
                "reply": "I'm here to listen and support you. Could you tell me more about how you're feeling?"
            }
        )

@app.get("/test")
async def test_page():
    """Serve the test HTML page"""
    try:
        with open("test_api.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Test page not found")

@app.post("/test-chat")
async def test_chat_endpoint(req: ChatRequest):
    """Test chat endpoint without authentication"""
    return ApiResponse(
        success=True,
        message="Test chat endpoint working",
        data={
            "message": req.message,
            "test": "This endpoint works without auth"
        }
    )

# -------------------------
# 🚀 Railway WSGI Application
# -------------------------

# This is the WSGI callable that Railway will use
application = app

if __name__ == "__main__":
    # For local development only
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")