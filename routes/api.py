from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import asyncio
import httpx
import json
from models.models import api_keys

router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/status")
async def api_status():
    # Return the status of the API and available services
    return {
        "openai": bool(api_keys["openai"]),
        "gemini": bool(api_keys["gemini"])
    }


@router.get("/test")
async def test_api():
    # Test the API connections and return diagnostic information
    results = {}
    
    # Test OpenAI API
    if api_keys["openai"]:
        try:
            import openai
            client = openai.OpenAI(api_key=api_keys["openai"])
            
            # Test with a simple prompt
            response = await asyncio.to_thread(
                lambda: client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Say hello in exactly 5 words."}],
                    max_tokens=20
                )
            )
            
            results["openai"] = {
                "status": "ok",
                "model": "gpt-3.5-turbo",
                "response": response.choices[0].message.content
            }
        except Exception as e:
            results["openai"] = {
                "status": "error",
                "error": str(e),
                "key_length": len(api_keys["openai"]) if api_keys["openai"] else 0
            }
    
    # Test Gemini API
    if api_keys["gemini"]:
        try:
            import google.generativeai as genai
            result = {"status": "testing"}
            
            # Initialize client
            genai.configure(api_key=api_keys["gemini"])
            
            # List available models
            result["step"] = "Listing available models"
            
            async def list_models():
                try:
                    models = genai.list_models()
                    return [model.name for model in models]
                except Exception as e:
                    return [f"Error listing models: {str(e)}"]
            
            result["available_models"] = await list_models()
            
            # Test with a simple prompt
            result["step"] = "Testing content generation with multiple model formats"
            
            # Try different model formats
            models_to_try = [
                "gemini-1.5-pro",  # Latest model
                "gemini-1.5-flash", # Faster model
                "gemini-pro",      # Older model
                "gemini-pro-vision" # Vision model
            ]
            
            result["model_tests"] = {}
            working_model = None
            
            async def test_model(model_name):
                try:
                    model = genai.GenerativeModel(model_name)
                    response = await model.generate_content_async(
                        "Say hello in exactly 5 words."
                    )
                    return {
                        "status": "success",
                        "response": response.text
                    }
                except Exception as e:
                    return {
                        "status": "error",
                        "error": str(e)
                    }
            
            for model in models_to_try:
                # Test this model
                model_result = await test_model(model)
                
                if model_result["status"] == "success":
                    result["model_tests"][model] = model_result
                    working_model = model
                    result["working_model"] = model
                    # Break once we find a working model
                    break
                else:
                    result["model_tests"][model] = model_result
            
            if working_model:
                result["test_response"] = result["model_tests"][working_model]["response"]
                result["status"] = "success"
            else:
                result["status"] = "error"
                result["error"] = "All model formats failed"
            
            results["gemini"] = result
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["key_length"] = len(api_keys["gemini"]) if api_keys["gemini"] else 0
            results["gemini"] = result
    
    return results

@router.get("/api/manual_test/{provider}")
async def manual_test(provider: str):
    # Run a manual test for a specific provider with detailed output
    if provider == "openai":
        if not api_keys["openai"]:
            return {"status": "error", "message": "No OpenAI API key provided"}
        
        try:
            import openai
            client = openai.OpenAI(api_key=api_keys["openai"])
            
            # Test with a simple prompt
            response = await asyncio.to_thread(
                lambda: client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Say hello in exactly 5 words."}],
                    max_tokens=20
                )
            )
            
            return {
                "status": "success",
                "model": "gpt-3.5-turbo",
                "response": response.choices[0].message.content,
                "full_response": json.loads(response.model_dump_json())
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "key_length": len(api_keys["openai"]) if api_keys["openai"] else 0
            }
    
    elif provider == "gemini":
        if not api_keys["gemini"]:
            return {"status": "error", "message": "No Gemini API key provided"}
        
        try:
            import google.generativeai as genai
            
            # Initialize client
            genai.configure(api_key=api_keys["gemini"])
            
            # Try with a working model (gemini-pro is most reliable)
            model_name = "gemini-pro"
            
            # Test with a simple prompt
            async def test_model():
                try:
                    model = genai.GenerativeModel(model_name)
                    response = await model.generate_content_async(
                        "Say hello in exactly 5 words."
                    )
                    return {
                        "status": "success",
                        "model": model_name,
                        "response": response.text,
                        "full_response": json.loads(json.dumps(response._raw_response))
                    }
                except Exception as e:
                    return {
                        "status": "error",
                        "error": str(e)
                    }
            
            return await test_model()
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "key_length": len(api_keys["gemini"]) if api_keys["gemini"] else 0
            }
    
    else:
        return {"status": "error", "message": f"Unsupported provider: {provider}"}