"""
LightGBM RRT Decision Support System - Backend API
Built with FastAPI RESTful API
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import os

from api.routes import router

# Select data loader based on environment variable
USE_DEMO_MODE = os.getenv("USE_DEMO_MODE", "true").lower() == "true"

if USE_DEMO_MODE:
    from services.demo_data_loader import demo_data_loader as data_loader
    print("[Demo Mode] Enabled")
else:
    from services.data_loader import data_loader
    print("[Real Data Mode] Enabled")

# ============================================================================
# Lifecycle management (recommended in modern FastAPI)
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Execute on startup
    print("\n=" * 60)
    print("[Startup] LightGBM RRT Decision Support API starting...")
    print("=" * 60)
    data_loader.load_all()
    print("[Done] All resources loaded")
    print("=" * 60 + "\n")
    
    yield
    
    # Execute on shutdown (cleanup if needed)
    print("\n[Shutdown] API shutting down...")

# Create FastAPI app
app = FastAPI(
    title="LightGBM RRT Decision Support API", 
    version="2.0.0",
    description="LightGBM-based Renal Replacement Therapy Decision Support System",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Root route (must be registered before router for priority)
# ============================================================================
@app.get("/", response_class=HTMLResponse)
async def root():
    """API root - returns a friendly HTML page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>IQL RRT Decision Support API</title>
        <meta charset="utf-8">
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }
            .info {
                background: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }
            .endpoint {
                background: #e8f5e9;
                padding: 10px;
                margin: 5px 0;
                border-left: 4px solid #4caf50;
            }
            a {
                color: #3498db;
                text-decoration: none;
            }
            a:hover {
                text-decoration: underline;
            }
            .button {
                display: inline-block;
                background: #3498db;
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
                margin: 10px 5px 0 0;
                text-decoration: none;
            }
            .button:hover {
                background: #2980b9;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>LightGBM RRT Decision Support API</h1>
            
            <div class="info">
                <strong>Version:</strong> 2.0.0<br>
                <strong>status:</strong> <span style="color: green;">[Runin]</span><br>
                <strong>description:</strong> LightGBM-based Renal Replacement Therapy Decision Support System
            </div>
            
            <h2>API </h2>
            <p> Full API and Test </p>
            <a href="/docs" class="button"> API </a>
            <a href="/redoc" class="button"> ReDoc </a>
            
            <h2>canuse </h2>
            <div class="endpoint">
                <strong>GET</strong> <code>/api/cases</code> - CaseColumnTable
            </div>
            <div class="endpoint">
                <strong>GET</strong> <code>/api/cases/{case_id}</code> - Case 
            </div>
            <div class="endpoint">
                <strong>POST</strong> <code>/api/predict</code> - GeneratePrediction
            </div>
            <div class="endpoint">
                <strong>POST</strong> <code>/api/explain</code> - Generate 
            </div>
            <div class="endpoint">
                <strong>POST</strong> <code>/api/llm_explain</code> - LLM 
            </div>
            <div class="endpoint">
                <strong>GET</strong> <code>/api/health</code> - 
            </div>
            
            <h2>JSON response</h2>
            <p>if needtoJSON </p>
            <a href="/api/root">/api/root (JSON)</a>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/api/root")
async def root_json():
    """API Path - JSON """
    return {
        "message": "LightGBM RRT Decision Support API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": [
            "/api/cases",
            "/api/predict",
            "/api/explain",
            "/api/llm_explain",
            "/api/health"
        ],
        "docs": "/docs",
        "redoc": "/redoc"
    }


# ============================================================================
# API in Pathafter 
# ============================================================================
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
