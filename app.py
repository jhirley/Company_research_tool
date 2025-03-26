import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Import routes
from routes.setup import router as setup_router
from routes.research import router as research_router
from routes.export import router as export_router
from routes.api import router as api_router

# Import models
from models.models import load_prompts

# Initialize FastAPI app
app = FastAPI(title="Company Research Tool")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Include routers
app.include_router(setup_router)
app.include_router(research_router)
app.include_router(export_router)
app.include_router(api_router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    # Initialize API keys with empty strings if they don't exist
    from models.models import api_keys
    if "openai" not in api_keys:
        api_keys["openai"] = ""
    if "gemini" not in api_keys:
        api_keys["gemini"] = ""
    
    # Load prompts on startup
    load_prompts()

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)