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
    import os
    import shutil
    from datetime import datetime
    
    # Create backups directory if it doesn't exist
    backup_dir = "prompts_backups"
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    # Get current timestamp for backup filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a backup of the entire prompt_categories dictionary
    import json
    try:
        # Convert the nested dictionary to a JSON-serializable format
        backup_data = {}
        for category_name, categories in prompt_categories.items():
            backup_data[category_name] = {}
            for subcategory_name, prompts in categories.items():
                backup_data[category_name][subcategory_name] = prompts
        
        # Save the backup to a JSON file
        backup_json = os.path.join(backup_dir, f"prompt_categories_{timestamp}.json")
        with open(backup_json, "w", encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not create JSON backup of prompt_categories: {e}")
    
    # Get the list of existing prompt files
    existing_files = [f for f in os.listdir() if f.startswith("Prompts_") and f.endswith(".txt")]
    
    # Create a mapping of existing files to the categories they contain
    file_to_categories = {}
    
    # First, load the content of each file to determine what categories it contains
    for file_name in existing_files:
        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extract categories from the file content
            categories_in_file = set()
            current_category = None
            
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("###"):
                    current_category = line.replace("###", "").strip()
                    categories_in_file.add(current_category)
            
            # Add this file to our mapping
            file_to_categories[file_name] = categories_in_file
        except Exception as e:
            print(f"Warning: Could not read categories from {file_name}: {e}")
    
    # Now create a reverse mapping from category to file
    category_to_file = {}
    
    # For each category in our prompt_categories
    for category_name in prompt_categories.keys():
        # Find which file contains this category
        for file_name, categories in file_to_categories.items():
            if category_name in categories:
                category_to_file[category_name] = file_name
                break
    
    # Now save each category to its corresponding file
    for category_name, categories in prompt_categories.items():
        # Determine file name - use existing file if available
        if category_name in category_to_file:
            file_name = category_to_file[category_name]
        else:
            file_name = f"Prompts_{category_name}.txt"
        
        # Create a backup of the original file if it exists
        if os.path.exists(file_name):
            backup_file = os.path.join(backup_dir, f"{file_name}.{timestamp}.bak")
            try:
                shutil.copy2(file_name, backup_file)
            except Exception as e:
                print(f"Warning: Could not create backup of {file_name}: {e}")
        
        try:
            with open(file_name, "w", encoding='utf-8') as file:
                for cat_name, prompts in categories.items():
                    # Sort prompts by number before writing
                    sorted_prompts = sorted(prompts, key=lambda x: x['number'])
                    
                    # Write subcategory header
                    file.write(f"### {cat_name}\n")
                    
                    # Write each prompt
                    for prompt in sorted_prompts:
                        file.write(f"{prompt['number']}. {prompt['text']}\n")
                    
                    # Add an empty line between subcategories for readability
                    file.write("\n")
            print(f"Successfully saved {file_name}")
        except Exception as e:
            print(f"Error saving {file_name}: {e}")

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