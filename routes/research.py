from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import asyncio
import re
from models.models import api_keys, current_companies, research_results, prompt_categories

router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/research", response_class=HTMLResponse)
async def research_page(request: Request):
    return templates.TemplateResponse(
        "research.html", 
        {
            "request": request, 
            "companies": current_companies,
            "prompt_categories": prompt_categories
        }
    )

@router.get("/prompts", response_class=HTMLResponse)
async def prompts_page(request: Request):
    return templates.TemplateResponse(
        "prompts.html", 
        {
            "request": request, 
            "prompt_categories": prompt_categories
        }
    )

@router.post("/update_prompts")
async def update_prompts(request: Request):
    form_data = await request.form()
    
    # Extract form data
    category = form_data.get("category")
    subcategory = form_data.get("subcategory")
    prompt_number = form_data.get("prompt_number")
    prompt_text = form_data.get("prompt_text")
    
    # Validate inputs
    if not all([category, subcategory, prompt_number, prompt_text]):
        return JSONResponse(
            status_code=400,
            content={"error": "All fields are required"}
        )
    
    try:
        prompt_number = int(prompt_number)
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"error": "Prompt number must be an integer"}
        )
    
    # Update prompt
    if category in prompt_categories and subcategory in prompt_categories[category]:
        for i, prompt in enumerate(prompt_categories[category][subcategory]):
            if prompt["number"] == prompt_number:
                prompt_categories[category][subcategory][i]["text"] = prompt_text
                break
        else:
            prompt_categories[category][subcategory].append({"number": prompt_number, "text": prompt_text})
    else:
        if category not in prompt_categories:
            prompt_categories[category] = {}
        if subcategory not in prompt_categories[category]:
            prompt_categories[category][subcategory] = []
        prompt_categories[category][subcategory].append({"number": prompt_number, "text": prompt_text})
    
    # Save prompts to file
    from models.models import save_prompts
    save_prompts()
    
    return JSONResponse(content={"success": True})

@router.post("/conduct_research")
async def conduct_research(
    request: Request,
    company: str = Form(...),
    prompt_category: str = Form(...),
    subcategory: str = Form(...),
    prompt_number: int = Form(...)
):
    # Validate inputs
    if not api_keys["openai"] and not api_keys["gemini"]:
        return JSONResponse(
            status_code=400,
            content={"error": "No API key provided. Please add an OpenAI or Gemini API key on the home page."}
        )
    
    if company not in current_companies:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid company '{company}'. Please select a valid company from the list."}
        )
    
    if prompt_category not in prompt_categories:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid category '{prompt_category}'. Please select a valid category."}
        )
    
    if subcategory not in prompt_categories[prompt_category]:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid subcategory '{subcategory}'. Please select a valid subcategory."}
        )
    
    # Find the prompt
    prompt_obj = None
    for p in prompt_categories[prompt_category][subcategory]:
        if p["number"] == prompt_number:
            prompt_obj = p
            break
    
    if not prompt_obj:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid prompt number '{prompt_number}'. Please select a valid prompt."}
        )
    
    # Replace placeholder with company name
    prompt = prompt_obj["text"].replace("{Customer}", company)
    prompt = prompt.replace("{Customer Name}", company)
    
    # Conduct research using the appropriate API
    result = ""
    sources = []
    
    try:
        print(f"Starting research for {company} using {'OpenAI' if api_keys['openai'] else 'Gemini'} API")
        if api_keys["openai"]:
            result, sources = await conduct_research_openai(prompt, company)
        elif api_keys["gemini"]:
            result, sources = await conduct_research_gemini(prompt, company)
    except Exception as e:
        error_message = str(e)
        print(f"Research error: {error_message}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Research failed: {error_message}"}
        )
    
    # Validate result
    if not result or not isinstance(result, str):
        return JSONResponse(
            status_code=500,
            content={"error": "Invalid response from AI service. Please try again."}
        )
    
    # Store research results
    if company not in research_results:
        research_results[company] = []
    
    # Check if this research already exists
    existing_entry = False
    for entry in research_results[company]:
        if (entry["category"] == prompt_category and 
            entry["subcategory"] == subcategory and 
            entry["prompt"] == prompt):
            existing_entry = True
            entry["result"] = result
            entry["sources"] = sources
            break
    
    if not existing_entry:
        research_results[company].append({
            "prompt": prompt,
            "result": result,
            "sources": sources,
            "category": prompt_category,
            "subcategory": subcategory
        })
    
    # Return the research results
    return {
        "company": company,
        "prompt": prompt,
        "result": result,
        "sources": sources
    }

@router.post("/conduct_research_all")
async def conduct_research_all(
    request: Request,
    company: str = Form(...)
):
    # Validate inputs
    if not api_keys["openai"] and not api_keys["gemini"]:
        return JSONResponse(
            status_code=400,
            content={"error": "No API key provided. Please add an OpenAI or Gemini API key on the home page."}
        )
    
    if company not in current_companies:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid company '{company}'. Please select a valid company from the list."}
        )
    
    # Initialize results storage if needed
    if company not in research_results:
        research_results[company] = []
    
    # Track all results
    all_results = []
    errors = []
    
    # Iterate through all categories, subcategories, and prompts
    for category in prompt_categories:
        for subcategory in prompt_categories[category]:
            for prompt_obj in prompt_categories[category][subcategory]:
                prompt_number = prompt_obj["number"]
                prompt_text = prompt_obj["text"]
                
                # Replace placeholder with company name
                prompt = prompt_text.replace("{Customer}", company)
                prompt = prompt.replace("{Customer Name}", company)
                
                try:
                    # Conduct research using the appropriate API
                    result = ""
                    sources = []
                    
                    print(f"Starting research for {company}, {category}/{subcategory}, prompt #{prompt_number}")
                    if api_keys["openai"]:
                        result, sources = await conduct_research_openai(prompt, company)
                    elif api_keys["gemini"]:
                        result, sources = await conduct_research_gemini(prompt, company)
                    
                    # Store research results (avoid duplicates)
                    existing_entry = False
                    for entry in research_results[company]:
                        if (entry["category"] == category and 
                            entry["subcategory"] == subcategory and 
                            entry["prompt"] == prompt):
                            existing_entry = True
                            entry["result"] = result
                            entry["sources"] = sources
                            break
                    
                    if not existing_entry:
                        research_results[company].append({
                            "prompt": prompt,
                            "result": result,
                            "sources": sources,
                            "category": category,
                            "subcategory": subcategory
                        })
                    
                    # Add to results for this batch
                    all_results.append({
                        "category": category,
                        "subcategory": subcategory,
                        "prompt_number": prompt_number,
                        "prompt": prompt,
                        "result": result,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_message = str(e)
                    print(f"Research error for {category}/{subcategory}, prompt #{prompt_number}: {error_message}")
                    errors.append({
                        "category": category,
                        "subcategory": subcategory,
                        "prompt_number": prompt_number,
                        "error": error_message
                    })
    
    # Return the research results
    return {
        "company": company,
        "results": all_results,
        "errors": errors,
        "total_successful": len(all_results),
        "total_errors": len(errors)
    }

# OpenAI research function
async def conduct_research_openai(prompt: str, company: str):
    from openai import AsyncOpenAI
    import json
    import logging
    import traceback
    
    try:
        print(f"Starting OpenAI research for company: {company}")
        print(f"Using API key: {api_keys['openai'][:5]}...{api_keys['openai'][-4:] if len(api_keys['openai']) > 10 else ''}")
        
        # Create a client instance with the API key
        client = AsyncOpenAI(api_key=api_keys["openai"])
        
        # Create a system message that emphasizes the need to avoid hallucinations
        system_message = f"""
        You are a professional business analyst researching {company}. 
        Your task is to provide accurate, factual information based on verifiable data.
        
        IMPORTANT GUIDELINES:
        1. Only provide information that you can verify from reliable sources.
        2. Clearly cite your sources for each piece of information.
        3. If you don't have enough information to answer a question, state this clearly rather than making assumptions.
        4. Avoid making predictions or speculations unless explicitly asked to do so.
        5. Format your response in a clear, professional manner.
        6. Include a "SOURCES" section at the end with numbered references to all sources used.
        """
        
        print(f"Sending prompt to OpenAI: {prompt[:50]}...")
        
        # Make the API call with the new client format
        try:
            response = await client.chat.completions.create(
                model="gpt-4",  # Using GPT-4 model (fallback from gpt-4o which might not be available)
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more factual responses
                max_tokens=2000
            )
            print("Successfully received response from OpenAI")
        except Exception as api_error:
            print(f"Error during OpenAI API call: {str(api_error)}")
            print(f"Error details: {traceback.format_exc()}")
            raise Exception(f"OpenAI API call failed: {str(api_error)}")
        
        # Extract the response text
        result = response.choices[0].message.content
        print(f"Received result of length: {len(result)} characters")
        
        # Extract sources from the response
        sources = []
        if "SOURCES" in result:
            try:
                sources_section = result.split("SOURCES")[1]
                source_lines = sources_section.strip().split("\n")
                for line in source_lines:
                    if line.strip():
                        # Remove any numbering or bullet points at the beginning
                        cleaned_line = re.sub(r'^[\d\.\-\*\s]+', '', line.strip())
                        sources.append(cleaned_line)
                print(f"Extracted {len(sources)} sources from the response")
            except Exception as source_error:
                print(f"Error extracting sources: {str(source_error)}")
        else:
            print("No SOURCES section found in the response")
        
        return result, sources
    except Exception as e:
        print(f"OpenAI API error: {str(e)}")
        print(f"Full error details: {traceback.format_exc()}")
        raise Exception(f"Error with OpenAI API: {str(e)}")

# Gemini research function
async def conduct_research_gemini(prompt: str, company: str):
    import google.generativeai as genai
    import asyncio
    import traceback
    
    try:
        print(f"Starting Gemini research for company: {company}")
        print(f"Using API key: {api_keys['gemini'][:5]}...{api_keys['gemini'][-4:] if len(api_keys['gemini']) > 10 else ''}")
        
        # Initialize the client with the API key
        try:
            genai.configure(api_key=api_keys["gemini"])
            print("Successfully initialized Gemini client")
        except Exception as client_error:
            print(f"Error initializing Gemini client: {str(client_error)}")
            print(f"Error details: {traceback.format_exc()}")
            raise Exception(f"Failed to initialize Gemini client: {str(client_error)}")
        
        # Create a system message that emphasizes the need to avoid hallucinations
        system_prompt = f"""
        You are a professional business analyst researching {company}. 
        Your task is to provide accurate, factual information based on verifiable data.
        
        IMPORTANT GUIDELINES:
        1. Only provide information that you can verify from reliable sources.
        2. Clearly cite your sources for each piece of information.
        3. If you don't have enough information to answer a question, state this clearly rather than making assumptions.
        4. Avoid making predictions or speculations unless explicitly asked to do so.
        5. Format your response in a clear, professional manner.
        6. Include a "SOURCES" section at the end with numbered references to all sources used.
        """
        
        print(f"Sending prompt to Gemini: {prompt[:50]}...")
        
        # Make the API call using synchronous code wrapped in asyncio.to_thread
        def generate_sync():
            try:
                # Try with different model names to find one that works
                models_to_try = [
                    "gemini-1.5-pro",
                    "gemini-1.5-flash",
                    "gemini-pro",
                    "gemini-pro-vision"
                ]
                
                last_error = None
                for model_name in models_to_try:
                    try:
                        print(f"Attempting to use Gemini model: {model_name}")
                        model = genai.GenerativeModel(model_name)
                        response = model.generate_content(
                            [system_prompt, prompt],
                            generation_config={"temperature": 0.3}  # Lower temperature for more factual responses
                        )
                        print(f"Successfully received response from Gemini model: {model_name}")
                        return response
                    except Exception as model_error:
                        last_error = model_error
                        print(f"Error with model {model_name}: {str(model_error)}")
                
                # If we get here, all models failed
                raise Exception(f"All Gemini models failed. Last error: {str(last_error)}")
            except Exception as api_error:
                print(f"Error during Gemini API call: {str(api_error)}")
                print(f"Error details: {traceback.format_exc()}")
                raise Exception(f"Gemini API call failed: {str(api_error)}")
        
        # Run the synchronous function in a thread pool
        response = await asyncio.to_thread(generate_sync)
        
        # Extract the response text
        try:
            result = response.text
            print(f"Received result of length: {len(result)} characters")
        except Exception as text_error:
            print(f"Error extracting text from Gemini response: {str(text_error)}")
            print(f"Response object: {response}")
            raise Exception(f"Failed to extract text from Gemini response: {str(text_error)}")
        
        # Extract sources from the response
        sources = []
        if "SOURCES" in result:
            try:
                sources_section = result.split("SOURCES")[1]
                source_lines = sources_section.strip().split("\n")
                for line in source_lines:
                    if line.strip():
                        # Remove any numbering or bullet points at the beginning
                        cleaned_line = re.sub(r'^[\d\.\-\*\s]+', '', line.strip())
                        sources.append(cleaned_line)
                print(f"Extracted {len(sources)} sources from the response")
            except Exception as source_error:
                print(f"Error extracting sources: {str(source_error)}")
        else:
            print("No SOURCES section found in the response")
        
        return result, sources
    except Exception as e:
        print(f"Gemini API error: {str(e)}")
        print(f"Full error details: {traceback.format_exc()}")
        raise Exception(f"Error with Gemini API: {str(e)}")