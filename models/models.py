import os
import re
import json

# Global variables to store API keys and settings
api_keys = {"openai": "", "gemini": ""}
current_companies = []
research_results = {}
prompt_categories = {}

# Load prompts from files
def load_prompts():
    global prompt_categories
    prompt_files = [f for f in os.listdir() if f.startswith("Prompts_") and f.endswith(".txt")]
    
    for file_name in prompt_files:
        # Try different encodings to handle potential encoding issues
        encodings = ['utf-8', 'latin-1', 'cp1252']
        content = None
        
        for encoding in encodings:
            try:
                with open(file_name, "r", encoding=encoding) as file:
                    content = file.read()
                    break  # If successful, break the loop
            except UnicodeDecodeError:
                continue  # Try the next encoding
        
        if content is None:
            print(f"Warning: Could not decode {file_name} with any of the attempted encodings.")
            continue
            
        category_name = file_name.replace("Prompts_", "").replace(".txt", "")
        categories = {}
        
        # Extract categories and their prompts
        current_category = None
        current_prompts = []
        
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a category header
            if line.startswith("###"):
                if current_category and current_prompts:
                    categories[current_category] = current_prompts
                current_category = line.replace("###", "").strip()
                current_prompts = []
            elif line.startswith("##"):
                if current_category and current_prompts:
                    categories[current_category] = current_prompts
                current_category = line.replace("##", "").strip()
                current_prompts = []
            elif line.startswith("#"):
                if current_category and current_prompts:
                    categories[current_category] = current_prompts
                current_category = line.replace("#", "").strip()
                current_prompts = []
            # Check if this is a numbered prompt
            elif re.match(r'^\d+\.', line) and current_category:
                # Extract the prompt number and text
                match = re.match(r'^(\d+)\.\s*(.*)', line)
                if match:
                    prompt_num = int(match.group(1))
                    prompt_text = match.group(2)
                    current_prompts.append({"number": prompt_num, "text": prompt_text})
        
        # Add the last category
        if current_category and current_prompts:
            categories[current_category] = current_prompts
            
        prompt_categories[category_name] = categories

# Save prompts to files
def save_prompts():
    for category_name, categories in prompt_categories.items():
        file_name = f"Prompts_{category_name}.txt"
        with open(file_name, "w", encoding='utf-8') as file:
            for cat_name, prompts in categories.items():
                file.write(f"### {cat_name}\n")
                for prompt in prompts:
                    file.write(f"{prompt['number']}. {prompt['text']}\n")

# Global variables to store API keys and settings
main_api_keys = {"openai": "", "gemini": ""}
main_current_companies = []
main_research_results = {}
main_prompt_categories = {}

# Load prompts from files
def main_load_prompts():
    global main_prompt_categories
    main_prompt_files = [f for f in os.listdir() if f.startswith("Prompts_") and f.endswith(".txt")]
    
    for file_name in main_prompt_files:
        # Try different encodings to handle potential encoding issues
        encodings = ['utf-8', 'latin-1', 'cp1252']
        content = None
        
        for encoding in encodings:
            try:
                with open(file_name, "r", encoding=encoding) as file:
                    content = file.read()
                    break  # If successful, break the loop
            except UnicodeDecodeError:
                continue  # Try the next encoding
        
        if content is None:
            print(f"Warning: Could not decode {file_name} with any of the attempted encodings.")
            continue
            
        category_name = file_name.replace("Prompts_", "").replace(".txt", "")
        categories = {}
        
        # Extract categories and their prompts
        current_category = None
        current_prompts = []
        
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a category header
            if line.startswith("###"):
                if current_category and current_prompts:
                    categories[current_category] = current_prompts
                current_category = line.replace("###", "").strip()
                current_prompts = []
            elif line.startswith("##"):
                if current_category and current_prompts:
                    categories[current_category] = current_prompts
                current_category = line.replace("##", "").strip()
                current_prompts = []
            elif line.startswith("#"):
                if current_category and current_prompts:
                    categories[current_category] = current_prompts
                current_category = line.replace("#", "").strip()
                current_prompts = []
            # Check if this is a numbered prompt
            elif re.match(r'^\d+\.', line) and current_category:
                # Extract the prompt number and text
                match = re.match(r'^(\d+)\.\s*(.*)', line)
                if match:
                    prompt_num = int(match.group(1))
                    prompt_text = match.group(2)
                    current_prompts.append({"number": prompt_num, "text": prompt_text})
        
        # Add the last category
        if current_category and current_prompts:
            categories[current_category] = current_prompts
            
        main_prompt_categories[category_name] = categories

# Save prompts to files
def main_save_prompts():
    for category_name, categories in main_prompt_categories.items():
        file_name = f"Prompts_{category_name}.txt"
        with open(file_name, "w", encoding='utf-8') as file:
            for cat_name, prompts in categories.items():
                file.write(f"### {cat_name}\n")
                for prompt in prompts:
                    file.write(f"{prompt['number']}. {prompt['text']}\n")