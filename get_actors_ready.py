import os
import re
import sys
import requests
from bs4 import BeautifulSoup
import config

def fetch_html_content(url):
    """
    Fetches HTML content from a given URL using requests.
    Returns the HTML text if the request is successful.
    """
    response = requests.get(url)
    response.raise_for_status()  # Raises HTTPError if status is 4xx or 5xx
    return response.text

def parse_cast_data(html_content, limit=10):
    """
    Parses the provided HTML content and returns up to `limit` tuples of:
       (actor_name, character_name, image_url).

    Looks for:
      - <h4> for the actor name
      - <p class="h4 unbold"> for the character name
      - <img> for the actor's image URL (preferring data-src-x2, else src)
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Find the full credits section
    full_credits_section = soup.find(id="fullcredits-content")
    if not full_credits_section:
        print("Could not find the #fullcredits-content section in the HTML.")
        return []
    
    # Each actor entry is often in a div with classes like "col-xs-12 col-md-6"
    actor_divs = full_credits_section.find_all("div", class_="col-xs-12 col-md-6")
    
    cast_data = []
    
    for div in actor_divs:
        # Actor name: <h4>Actor Name</h4>
        actor_name_tag = div.find("h4")
        if not actor_name_tag:
            continue
        
        actor_name = actor_name_tag.get_text(strip=True)
        
        # Character name: <p class="h4 unbold">Character Name (episodes...)</p>
        character_tag = div.find("p", class_="h4 unbold")
        if not character_tag:
            continue
        
        # Example text: "Ted Mosby (208 episodes, 2005-2014)"
        # We split by '(' to remove episode info => "Ted Mosby"
        character_full = character_tag.get_text(strip=True)
        character_name = character_full.split("(")[0].strip()
        
        # Image URL: prefer data-src-x2; otherwise, src
        img_tag = div.find("img")
        if not img_tag:
            continue
        image_url = img_tag.get("data-src-x2") or img_tag.get("src")
        
        cast_data.append((actor_name, character_name, image_url))
        
        # Stop if we reach the limit
        if len(cast_data) == limit:
            break
    
    return cast_data

def sanitize_for_filename(text):
    """
    Removes or replaces characters that are unsafe for Windows filenames,
    including newlines, carriage returns, punctuation, etc.
    Returns a 'safe' string.
    """
    # Replace newlines
    text = text.replace("\n", " ").replace("\r", " ")
    # Remove characters not alphanumeric, underscore, space, or hyphen
    text = re.sub(r"[^\w\s-]", "", text)
    # Convert multiple spaces to a single underscore
    text = re.sub(r"\s+", "_", text.strip())
    return text

def download_images(cast_data, output_folder="actors"):
    """
    Downloads images for each tuple in cast_data (actor_name, character_name, image_url).

    - Files are named "ActorName_as_CharacterName.jpg".
    - Each file is saved in a subfolder with the same name as the file (minus extension).
      So, for file "Josh_Radnor_as_Ted_Mosby.jpg", the path is:
         actors/Josh_Radnor_as_Ted_Mosby/Josh_Radnor_as_Ted_Mosby.jpg

    If any error (like invalid file name) occurs, we skip that entry.
    """
    os.makedirs(output_folder, exist_ok=True)  # Base folder, e.g. "actors"
    
    for i, (actor_name, character_name, image_url) in enumerate(cast_data, start=1):
        if not image_url:
            continue
        
        # Sanitize names for the filesystem
        safe_actor_name = sanitize_for_filename(actor_name)
        safe_character_name = sanitize_for_filename(character_name)
        
        # Construct a file name, e.g. "Josh_Radnor_as_Ted_Mosby.jpg"
        file_name = f"{safe_actor_name}_as_{safe_character_name}.jpg"
        
        # Make a subfolder named the same as the file (without .jpg)
        # e.g., subfolder "Josh_Radnor_as_Ted_Mosby"
        subfolder_name = file_name.rsplit(".", 1)[0]  # remove .jpg extension
        subfolder_path = os.path.join(output_folder, subfolder_name)
        
        os.makedirs(subfolder_path, exist_ok=True)
        
        # Final path: "actors/Josh_Radnor_as_Ted_Mosby/Josh_Radnor_as_Ted_Mosby.jpg"
        file_path = os.path.join(subfolder_path, file_name)
        
        print(f"Downloading image {i} for: {actor_name} as {character_name}")
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            
            with open(file_path, "wb") as f:
                f.write(response.content)
        except (requests.RequestException, OSError) as e:
            print(f"Skipping '{actor_name}' as '{character_name}' due to error:\n  {e}")

def download_imdb_cast_images(imdb_cast_url=None, limit=None):
    """
    High-level function to:
    1. Fetch HTML from the IMDb cast URL.
    2. Parse (actor_name, character_name, image_url).
    3. Download and name each image in actor-specific subfolders within data directory.
    """
    # Get URL from user if not provided
    if imdb_cast_url is None:
        imdb_cast_url = input("Please enter the IMDb cast URL: ")
    
    # Get limit from user if not provided
    if limit is None:
        limit = int(input("How many actors do you want to download? "))
    
    # 1. Fetch HTML
    html_content = fetch_html_content(imdb_cast_url)
    
    # 2. Parse up to `limit` entries
    cast_data = parse_cast_data(html_content, limit=limit)
    if not cast_data:
        print("No cast data found.")
        return
    
    # Create actors directory inside the existing data folder
    main_folder = os.path.join("data", "actors")
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
        print(f"Created directory: {main_folder}")
    
    # Process each actor
    for i, (actor_name, character_name, image_url) in enumerate(cast_data, start=1):
        if not image_url:
            continue
        
        # Create sanitized folder name from actor and character names
        safe_actor_name = sanitize_for_filename(actor_name)
        safe_character_name = sanitize_for_filename(character_name)
        folder_name = f"{safe_actor_name}_as_{safe_character_name}"
        folder_path = os.path.join(main_folder, folder_name)
        
        # Skip if folder already exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Download image with same name as folder
        file_name = f"{folder_name}.jpg"
        file_path = os.path.join(folder_path, file_name)
        
        print(f"Downloading image {i} for: {actor_name} as {character_name}")
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            
            # Attempt to write the file
            with open(file_path, "wb") as f:
                f.write(response.content)
        except (requests.RequestException, OSError) as e:
            print(f"Skipping {actor_name} as {character_name} due to error:\n  {e}")

if __name__ == "__main__":
    # Example usage - now parameters will be requested from user
    download_imdb_cast_images()
