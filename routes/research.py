from fastapi import APIRouter, Request, Form, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import asyncio
import re
import logging
from models.models import api_keys, current_companies, research_results, prompt_categories

logger = logging.getLogger(__name__)

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
    
    # Process all form data to update prompts
    updated_prompts = {}
    
    # Iterate through all form fields
    for key, value in form_data.items():
        if key.startswith("prompt_"):
            # Extract category, subcategory, and prompt number from the field name
            # Format: prompt_CATEGORY_SUBCATEGORY_NUMBER
            parts = key.split("_", 1)[1].rsplit("_", 1)
            prompt_number = int(parts[1])
            cat_subcat = parts[0].rsplit("_", 1)
            
            if len(cat_subcat) == 2:
                category, subcategory = cat_subcat
            else:
                # Handle case where subcategory name might contain underscores
                category = cat_subcat[0]
                subcategory = "_".join(cat_subcat[1:])
            
            # Initialize nested dictionary structure if needed
            if category not in updated_prompts:
                updated_prompts[category] = {}
            if subcategory not in updated_prompts[category]:
                updated_prompts[category][subcategory] = []
                
            # Add the prompt to the updated structure
            updated_prompts[category][subcategory].append({
                "number": prompt_number,
                "text": value
            })
    
    # Update the global prompt_categories with the new data
    # This ensures we don't lose any categories or subcategories
    for category, subcategories in updated_prompts.items():
        if category not in prompt_categories:
            prompt_categories[category] = {}
            
        for subcategory, prompts in subcategories.items():
            # Replace the entire list of prompts for this subcategory
            prompt_categories[category][subcategory] = prompts
    
    # Save prompts to file
    from models.models import save_prompts
    save_prompts()
    
    # Redirect to the prompts page instead of returning JSON
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/prompts", status_code=303)

@router.post("/add_category")
async def add_category(request: Request):
    data = await request.json()
    category_name = data.get("category_name")
    
    if not category_name:
        return JSONResponse(
            status_code=400,
            content={"error": "Category name is required"}
        )
    
    # Convert spaces to underscores for consistency
    category_name = category_name.replace(" ", "_")
    
    # Check if category already exists
    if category_name in prompt_categories:
        return JSONResponse(
            status_code=400,
            content={"error": "Category already exists"}
        )
    
    # Create new category
    prompt_categories[category_name] = {}
    
    # Save prompts to file
    from models.models import save_prompts
    save_prompts()
    
    return JSONResponse(content={"success": True})

@router.post("/add_subcategory")
async def add_subcategory(request: Request):
    data = await request.json()
    category = data.get("category")
    subcategory_name = data.get("subcategory_name")
    
    if not all([category, subcategory_name]):
        return JSONResponse(
            status_code=400,
            content={"error": "Category and subcategory name are required"}
        )
    
    # Check if category exists
    if category not in prompt_categories:
        return JSONResponse(
            status_code=400,
            content={"error": "Category does not exist"}
        )
    
    # Check if subcategory already exists
    if subcategory_name in prompt_categories[category]:
        return JSONResponse(
            status_code=400,
            content={"error": "Subcategory already exists in this category"}
        )
    
    # Create new subcategory
    prompt_categories[category][subcategory_name] = []
    
    # Save prompts to file
    from models.models import save_prompts
    save_prompts()
    
    return JSONResponse(content={"success": True})

@router.post("/add_prompt")
async def add_prompt(request: Request):
    data = await request.json()
    category = data.get("category")
    subcategory = data.get("subcategory")
    prompt_number = data.get("prompt_number")
    prompt_text = data.get("prompt_text")
    
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
    
    # Check if category and subcategory exist
    if category not in prompt_categories:
        return JSONResponse(
            status_code=400,
            content={"error": "Category does not exist"}
        )
    
    if subcategory not in prompt_categories[category]:
        return JSONResponse(
            status_code=400,
            content={"error": "Subcategory does not exist in this category"}
        )
    
    # Check if prompt number already exists
    for prompt in prompt_categories[category][subcategory]:
        if prompt["number"] == prompt_number:
            return JSONResponse(
                status_code=400,
                content={"error": "A prompt with this number already exists in this subcategory"}
            )
    
    # Add new prompt
    prompt_categories[category][subcategory].append({
        "number": prompt_number,
        "text": prompt_text
    })
    
    # Sort prompts by number
    prompt_categories[category][subcategory].sort(key=lambda x: x["number"])
    
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
    if not api_keys.get("openai") and not api_keys.get("gemini"):
        return JSONResponse(
            status_code=400,
            content={"error": "No API key provided. Please add an OpenAI or Gemini API key on the home page."}
        )
        
    # Validate API keys format
    if api_keys.get("openai") and len(api_keys["openai"]) < 20:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid OpenAI API key format. Please check your API key on the home page."}
        )
        
    if api_keys.get("gemini") and len(api_keys["gemini"]) < 20:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid Gemini API key format. Please check your API key on the home page."}
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
        print(f"Starting research for {company} using {'OpenAI' if api_keys.get('openai') else 'Gemini'} API")
        
        # Set a timeout for the research to prevent hanging
        try:
            if api_keys.get("openai") and len(api_keys["openai"]) >= 20:
                # Use asyncio.wait_for to set a timeout
                result, sources = await asyncio.wait_for(
                    conduct_research_openai(prompt, company),
                    timeout=60  # 60 second timeout
                )
            elif api_keys.get("gemini") and len(api_keys["gemini"]) >= 20:
                # Use asyncio.wait_for to set a timeout
                result, sources = await asyncio.wait_for(
                    conduct_research_gemini(prompt, company),
                    timeout=60  # 60 second timeout
                )
            else:
                raise Exception("No valid API key available")
        except asyncio.TimeoutError:
            raise Exception("Research timed out after 60 seconds. Please try again.")
            
    except Exception as e:
        error_message = str(e)
        print(f"Research error: {error_message}")
        
        # Check for network errors and provide a more user-friendly message
        if "Network" in error_message or "timeout" in error_message.lower() or "connection" in error_message.lower():
            user_message = "NetworkError when attempting to fetch resource. Please check your internet connection and try again."
        elif "key" in error_message.lower() or "auth" in error_message.lower() or "credential" in error_message.lower():
            user_message = "API key error. Please check your API key and try again."
        else:
            user_message = f"Research failed: {error_message}"
            
        return JSONResponse(
            status_code=500,
            content={"error": user_message}
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
        # Check for both old and new key names for backward compatibility
        entry_category = entry.get("promptCategory", entry.get("category", ""))
        if (entry_category == prompt_category and 
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
            "promptCategory": prompt_category,
            "subcategory": subcategory
        })
    
    # Log the current state of research_results for debugging
    logger.info(f"Updated research_results for {company}. Now has {len(research_results[company])} entries.")
    
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
    # This function now implements a streaming response pattern to prevent timeouts
    # Validate inputs
    if not api_keys.get("openai") and not api_keys.get("gemini"):
        return JSONResponse(
            status_code=400,
            content={"error": "No API key provided. Please add an OpenAI or Gemini API key on the home page."}
        )
        
    # Validate API keys format
    if api_keys.get("openai") and len(api_keys["openai"]) < 20:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid OpenAI API key format. Please check your API key on the home page."}
        )
        
    if api_keys.get("gemini") and len(api_keys["gemini"]) < 20:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid Gemini API key format. Please check your API key on the home page."}
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
    
    # Gather all research tasks to run sequentially
    research_tasks = []
    
    # First, collect all the research tasks with their metadata
    for category in prompt_categories:
        for subcategory in prompt_categories[category]:
            for prompt_obj in prompt_categories[category][subcategory]:
                prompt_number = prompt_obj["number"]
                prompt_text = prompt_obj["text"]
                
                # Replace placeholder with company name
                prompt = prompt_text.replace("{Customer}", company)
                prompt = prompt.replace("{Customer Name}", company)
                
                # Store task info
                research_tasks.append({
                    "category": category,
                    "subcategory": subcategory,
                    "prompt_number": prompt_number,
                    "prompt": prompt
                })
                
                print(f"Prepared research task for {company}, {category}/{subcategory}, prompt #{prompt_number}")
    
    print(f"Gathered {len(research_tasks)} research tasks to run")
    
    # Define a function to process a single task
    async def process_single_task(task_info):
        category = task_info["category"]
        subcategory = task_info["subcategory"]
        prompt_number = task_info["prompt_number"]
        prompt = task_info["prompt"]
        
        try:
            # Execute the task
            if api_keys.get("openai") and len(api_keys["openai"]) >= 20:
                result, sources = await conduct_research_openai(prompt, company)
            elif api_keys.get("gemini") and len(api_keys["gemini"]) >= 20:
                result, sources = await conduct_research_gemini(prompt, company)
            else:
                raise Exception("No valid API key available")
                
            print(f"Completed research for {company}, {category}/{subcategory}, prompt #{prompt_number}")
            
            # Return the result
            return {
                "success": True,
                "category": category,
                "subcategory": subcategory,
                "prompt_number": prompt_number,
                "prompt": prompt,
                "result": result,
                "sources": sources
            }
        except Exception as e:
            error_message = str(e)
            print(f"Research error for {company}, {category}/{subcategory}, prompt #{prompt_number}: {error_message}")
            
            # Check for network errors and provide a more user-friendly message
            if "Network" in error_message or "timeout" in error_message.lower() or "connection" in error_message.lower():
                error_message = "NetworkError when attempting to fetch resource. Please check your internet connection and try again."
            elif "key" in error_message.lower() or "auth" in error_message.lower() or "credential" in error_message.lower():
                error_message = "API key error. Please check your API key and try again."
            
            # Return the error
            return {
                "success": False,
                "category": category,
                "subcategory": subcategory,
                "prompt_number": prompt_number,
                "error": error_message
            }
    
    # Process tasks in parallel with concurrency control
    import asyncio
    from starlette.responses import StreamingResponse
    import json
    import time
    
    # Set maximum concurrent tasks (adjust based on your API rate limits)
    max_concurrent = 3
    
    # Create an async generator to stream results back to the client
    async def stream_results():
        task_results = []
        consecutive_errors = 0
        completed_count = 0
        total_tasks = len(research_tasks)
        
        # Send initial progress update
        yield json.dumps({
            "type": "progress",
            "message": f"Starting research for {company}",
            "completed": 0,
            "total": total_tasks,
            "percent": 0
        }) + "\n"
        
        # Process tasks in batches
        for i in range(0, total_tasks, max_concurrent):
            # Get the next batch of tasks
            batch = research_tasks[i:i+max_concurrent]
            batch_tasks = [process_single_task(task_info) for task_info in batch]
            
            # Wait for all tasks in this batch to complete
            batch_results = await asyncio.gather(*batch_tasks)
            
            # Process batch results
            batch_success = False
            batch_successes = []
            batch_errors = []
            
            for result in batch_results:
                completed_count += 1
                percent_complete = int((completed_count / total_tasks) * 100)
                
                if result["success"]:
                    batch_success = True
                    task_results.append(result)
                    batch_successes.append(result)
                    
                    # Store successful research results (avoid duplicates)
                    category = result["category"]
                    subcategory = result["subcategory"]
                    prompt = result["prompt"]
                    
                    existing_entry = False
                    for entry in research_results[company]:
                        # Check both formats for compatibility
                        entry_category = entry.get("category", entry.get("promptCategory", ""))
                        entry_subcategory = entry.get("subcategory", "")
                        
                        if (entry_category == category and 
                            entry_subcategory == subcategory and 
                            entry.get("prompt") == prompt):
                            existing_entry = True
                            entry["result"] = result["result"]
                            entry["sources"] = result["sources"]
                            # Ensure consistent keys
                            entry["category"] = category
                            entry["subcategory"] = subcategory
                            entry["promptCategory"] = category  # For backward compatibility
                            break
                    
                    if not existing_entry:
                        research_results[company].append({
                            "prompt": prompt,
                            "result": result["result"],
                            "sources": result["sources"],
                            "category": category,
                            "subcategory": subcategory,
                            "promptCategory": category  # For backward compatibility
                        })
                    
                    # Add to results for this batch
                    all_results.append({
                        "category": category,
                        "subcategory": subcategory,
                        "prompt_number": result["prompt_number"],
                        "prompt": prompt,
                        "result": result["result"],
                        "sources": result["sources"]
                    })
                else:
                    batch_errors.append({
                        "category": result["category"],
                        "subcategory": result["subcategory"],
                        "prompt_number": result["prompt_number"],
                        "error": result["error"]
                    })
                    errors.append({
                        "category": result["category"],
                        "subcategory": result["subcategory"],
                        "prompt_number": result["prompt_number"],
                        "error": result["error"]
                    })
            
            # Send progress update after each batch
            yield json.dumps({
                "type": "progress",
                "message": f"Completed {completed_count} of {total_tasks} tasks",
                "completed": completed_count,
                "total": total_tasks,
                "percent": percent_complete,
                "batch_results": {
                    "successes": [{
                        "category": r["category"],
                        "subcategory": r["subcategory"],
                        "prompt_number": r["prompt_number"]
                    } for r in batch_successes],
                    "errors": batch_errors
                }
            }) + "\n"
            
            # Check for consecutive errors
            if not batch_success:
                consecutive_errors += 1
            else:
                consecutive_errors = 0
            
            # If we've had 3 consecutive batches with all errors, stop the research
            if consecutive_errors >= 3 and len(task_results) == 0:
                print("Too many consecutive error batches, stopping research")
                yield json.dumps({
                    "type": "error",
                    "error": "Multiple research errors occurred. Please check your API key and internet connection.",
                    "details": errors
                }) + "\n"
                return
            
            # Save research results to backup file after each batch
            try:
                from routes.export import save_research_results
                save_research_results()
                print(f"Saved research results to backup after batch")
            except Exception as save_error:
                print(f"Error saving research results after batch: {save_error}")
            
            # Add a small delay to allow the client to process the update
            await asyncio.sleep(0.1)
        
        # Send final results
        yield json.dumps({
            "type": "complete",
            "results": all_results,
            "errors": errors,
            "total_successful": len(all_results),
            "total_errors": len(errors)
        }) + "\n"
    
    # Return a streaming response
    return StreamingResponse(
        stream_results(),
        media_type="text/event-stream"
    )
    
    # Save research results to backup file
    try:
        from routes.export import save_research_results
        save_research_results()
        print(f"Saved all research results to backup file")
    except Exception as save_error:
        print(f"Error saving research results to backup: {save_error}")
    
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
        6. Include a "SOURCES" section at the end with numbered references to all sources used with the source URL.
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
        6. Include a "SOURCES" section at the end with numbered references to all sources used with the source URL.
        """
        
        print(f"Sending prompt to Gemini: {prompt[:50]}...")
        
        # Make the API call using proper async handling
        async def try_models():
            # Try with different model names to find one that works
            models_to_try = [
                # "gemini-2.0-flash",
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "gemini-pro",
                "gemini-pro-vision"
            ]
            
            last_error = None
            for model_name in models_to_try:
                try:
                    print(f"Attempting to use Gemini model: {model_name}")
                    
                    # Create the model
                    model = genai.GenerativeModel(model_name)
                    
                    # Use the asynchronous API
                    response = await model.generate_content_async(
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
        
        # Call the async function
        response = await try_models()
        
        # Extract the response text
        try:
            # Access the text from the response
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


@router.post("/sync_research_history")
async def sync_research_history(data: dict = Body(...)):
    """Sync the frontend research history with the backend research_results"""
    company = data.get("company")
    history = data.get("history")
    
    logger.info(f"Syncing research history for company: {company}")
    logger.info(f"Received {len(history) if history else 0} history items")
    
    if not company or not history:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing company or history data"}
        )
    
    # Update the backend research_results with the frontend history
    research_results[company] = history
    
    # Save the updated research results to the backup file
    try:
        from routes.export import save_research_results
        save_research_results()
    except ImportError:
        # If the export module hasn't been imported yet, save directly
        import json
        with open("research_results_backup.json", 'w') as f:
            json.dump(research_results, f)
    
    logger.info(f"Successfully synced research history for {company}. Now has {len(research_results[company])} entries.")
    return JSONResponse(content={"status": "success"})