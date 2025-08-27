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
    from pydantic import BaseModel, field_validator
    from langchain_core.prompts import PromptTemplate
    from langchain_cohere import ChatCohere
    from supabase import create_client
    import traceback
    import uuid
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
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

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
    user_id: str = "anonymous"  # Default to anonymous if not provided
    
    @field_validator('message')
    @classmethod
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

# Initialize Supabase client
supabase = None

def initialize_supabase():
    """Initialize Supabase client"""
    global supabase
    
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        logger.warning("Supabase credentials not provided - logging disabled")
        return False
    
    try:
        global supabase
        supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        logger.info("Supabase client initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Supabase: {e}")
        return False

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
            model="command", 
            temperature=0.1, 
            max_tokens=150
        )
        
        emotion_prompt = PromptTemplate.from_template("""
You are an expert emotion classifier. Classify the message into exactly one emotion: joy, sadness, anger, fear, disgust, neutral.

SADNESS includes: sad, depressed, hopeless, lonely, down, miserable, heartbroken, grieving, disappointed, empty, worthless, suicidal thoughts, "I can't go on", "life is meaningless"

FEAR includes: scared, anxious, worried, terrified, nervous, stressed, panic, overwhelmed, but NOT depression or sadness

Message: {message}

Respond with ONLY one word: joy, sadness, anger, fear, disgust, or neutral
""")

        response_prompt = PromptTemplate.from_template("""
You are InsideOut, a warm and empathetic AI companion. Keep responses SHORT - maximum 3 sentences only.

User's message: {message}
Detected emotion: {emotion}

RESPONSE RULES:
- MAXIMUM 5 SENTENCES ONLY
- Use 1-2 emojis maximum
- Be warm and caring and offer to help.

For each emotion:
JOY: "I'm so happy for you! 😊 [1-2 sentences sharing their joy]"
SADNESS: "I'm really sorry you're feeling this way. 💙 [1-2 sentences offering comfort]"
ANGER: "I understand why you'd feel angry about that. 😤 [1-2 sentences validating feelings]"
FEAR: "That sounds really scary, and it's okay to feel afraid. 😰 [1-2 sentences offering reassurance]"
DISGUST: "That's definitely upsetting. 😣 [1-2 sentences validating reaction]"
NEUTRAL: "How are you really feeling today? 🤔 [1-2 sentences inviting sharing]"

CRISIS: If suicidal/self-harm thoughts: "I'm really worried about you. 💙 Please call Egypt's mental health hotlines: 16328, 105, or 15335, or visit https://mentalhealth.mohp.gov.eg/. You're not alone and things will get better."

Keep it SHORT and caring.
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

async def handle_registration(req: ChatRequest):
    """Handle user registration through chat endpoint"""
    global supabase
    
    # Extract username from message (format: "register:username")
    try:
        username = req.message.split(":", 1)[1].strip()
        if not username:
            return ApiResponse(
                success=False,
                message="Invalid registration",
                error="Please provide a username after 'register:'"
            )
    except IndexError:
        return ApiResponse(
            success=False,
            message="Invalid registration format",
            error="Use format: 'register:your_username'"
        )
    
    # Check if user already exists
    user_exists = False
    if supabase:
        try:
            user_uuid = generate_user_uuid(username)
            result = supabase.table("emotion_logs").select("user_id").eq("user_id", user_uuid).limit(1).execute()
            user_exists = len(result.data) > 0
        except Exception as e:
            logger.error(f"Failed to check user existence: {e}")
    
    # Log the registration
    await log_to_supabase(username, "REGISTRATION", "neutral", "User registered")
    
    if user_exists:
        return ApiResponse(
            success=True,
            message="Welcome back!",
            data={
                "user_id": username,
                "is_new_user": False,
                "emotion": "neutral",
                "reply": f"Welcome back, {username}! I'm so glad to see you again. How are you feeling today? 💙"
            }
        )
    else:
        return ApiResponse(
            success=True,
            message="Registration successful",
            data={
                "user_id": username,
                "is_new_user": True,
                "emotion": "neutral",
                "reply": f"Welcome, {username}! I'm InsideOut, your empathetic AI companion. I'm here to listen and support you. How are you feeling today? 💙"
            }
        )

def generate_user_uuid(username: str) -> str:
    """Generate a consistent UUID for a username"""
    # Use a deterministic UUID based on the username
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, username))

async def log_to_supabase(user_id: str, message: str, emotion: str, reply: str):
    """Log chat interaction to Supabase"""
    global supabase
    
    if not supabase:
        logger.warning("Supabase not initialized - skipping log")
        return
    
    try:
        # Generate UUID for user_id
        user_uuid = generate_user_uuid(user_id)
        
        # Insert data matching your table schema
        data = {
            "user_id": user_uuid,
            "message": message,
            "emotion": emotion,
            "created_at": datetime.utcnow().isoformat()
        }
        
        result = supabase.table("emotion_logs").insert(data).execute()
        logger.info(f"Logged interaction to Supabase: {result}")
        
    except Exception as e:
        logger.error(f"Failed to log to Supabase: {e}")

# Initialize AI on startup
initialize_ai()

# -------------------------
# 🚀 Application Startup
# -------------------------

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting InsideOut API...")
    
    # Initialize Supabase
    if initialize_supabase():
        logger.info("✅ Supabase initialized")
    else:
        logger.warning("⚠️ Supabase failed to initialize - logging will be disabled")
    
    # Initialize AI
    if initialize_ai():
        logger.info("✅ AI services initialized")
    else:
        logger.warning("⚠️ AI services failed to initialize - chat features will be unavailable")
    
    # Check environment variables
    missing_vars = []
    if not COHERE_API_KEY:
        missing_vars.append("COHERE_API_KEY")
    if not SUPABASE_URL:
        missing_vars.append("SUPABASE_URL")
    if not SUPABASE_ANON_KEY:
        missing_vars.append("SUPABASE_ANON_KEY")
    
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
    """Main chat endpoint with automatic user registration and crash prevention"""
    # Debug logging
    logger.info(f"Chat endpoint called. AI_AVAILABLE: {AI_AVAILABLE}, emotion_chain: {emotion_chain is not None}, response_chain: {response_chain is not None}")
    
    # Check if this is a registration request (special message format)
    if req.message.lower().startswith("register:"):
        return await handle_registration(req)
    
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
        
        # Get emotion detection (simplified without asyncio)
        detected_emotion = 'neutral'
        try:
            # Direct call to emotion chain
            emotion_result = emotion_chain.invoke({"message": req.message})
            
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
                
        except Exception as e:
            logger.error(f"Emotion detection failed: {e}")
            detected_emotion = 'neutral'
        
        # Generate response (simplified without asyncio)
        reply = "I'm here to listen and support you. Could you tell me more about how you're feeling?"
        try:
            start_time = time.time()
            # Direct call to response chain
            bot_response = response_chain.invoke({
                "emotion": detected_emotion,
                "message": req.message
            })
            response_time = time.time() - start_time
            logger.info(f"Response generated in {response_time:.2f} seconds")
            reply = bot_response.content.strip()
            
            # Validate response
            if not reply or len(reply.strip()) == 0:
                reply = "I'm here to listen and support you. Could you tell me more about how you're feeling?"
            
            # Limit response length
            if len(reply) > 500:
                reply = reply[:500] + "..."
                
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            reply = "I'm here to listen and support you. Could you tell me more about how you're feeling?"
        
        # Log to Supabase
        await log_to_supabase(req.user_id, req.message, detected_emotion, reply)
        
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

@app.get("/check-user/{user_id}")
async def check_user_exists(user_id: str):
    """Check if a user ID exists in the database"""
    global supabase
    
    if not supabase:
        return ApiResponse(
            success=False,
            message="Database not available",
            error="Supabase connection not available"
        )
    
    try:
        # Query the emotion_logs table for the user_id
        user_uuid = generate_user_uuid(user_id)
        result = supabase.table("emotion_logs").select("user_id").eq("user_id", user_uuid).limit(1).execute()
        
        user_exists = len(result.data) > 0
        
        return ApiResponse(
            success=True,
            message="User check completed",
            data={
                "user_id": user_id,
                "exists": user_exists,
                "message": f"Welcome back, {user_id}!" if user_exists else f"Welcome, {user_id}! This is your first time here."
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to check user existence: {e}")
        return ApiResponse(
            success=False,
            message="Failed to check user",
            error="Database error occurred"
        )

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