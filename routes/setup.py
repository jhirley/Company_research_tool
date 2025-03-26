from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from models.models import api_keys, current_companies

router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request, 
            "api_keys": api_keys,
            "companies": current_companies
        }
    )

@router.post("/setup")
async def setup(
    request: Request,
    openai_key: str = Form(None),
    gemini_key: str = Form(None),
    companies: str = Form(...)
):
    # Update API keys
    if openai_key:
        api_keys["openai"] = openai_key
    if gemini_key:
        api_keys["gemini"] = gemini_key
    
    # Update companies list
    company_list = [c.strip() for c in companies.split(",") if c.strip()]
    current_companies.clear()
    current_companies.extend(company_list)
    
    # Redirect to research page
    return RedirectResponse(url="/research", status_code=303)