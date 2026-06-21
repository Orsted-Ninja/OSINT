import subprocess
from langchain.tools import tool

@tool
def username_osint(username: str):
    """
    Perform OSINT username enumeration using Sherlock.

    This tool searches for a given username across multiple public websites
    to identify where the username is registered or in use.

    Input:
    - username (str): target username/handle (without quotes or @)

    Returns:
    - username (str): input username
    - results (list): up to 20 discovered profiles containing:
        - site (str): platform name (e.g., GitHub, Reddit, Twitter)
        - url (str): profile URL where username was found
    - count (int): total number of matches found
    - error (str, optional): error message if execution fails

    Use when:
    - user provides a username or online handle
    - mapping digital footprint across platforms
    - linking identities across social/media services

    Notes:
    - Uses Sherlock CLI via subprocess
    - Output is parsed from CLI text
    - Limited to 20 results to avoid large LLM payloads
    - Requires Sherlock installed and accessible in system PATH
    """
    try:
        import os
        from backend.core.proxy_config import PROXY
        env = os.environ.copy()
        env.update(PROXY.as_env_vars())

        result = subprocess.run(
            ["sherlock", username, "--output", os.devnull],
            capture_output=True,
            text=True,
            timeout=180,
            env=env
        )

        lines = result.stdout.splitlines()

        found = []
        for line in lines:
            if "[+]" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    found.append({
                        "site": parts[0].replace("[+]", "").strip(),
                        "url": parts[1].strip()
                    })
        from backend.agent_workflow.tools.image_extraction_tool import extract_images
        from backend.agent_workflow.tools.profile_scraper import scrape_profile
        
        # High-value platforms to auto-scrape for images and text
        high_value = {"GitHub", "AllMyLinks", "Medium", "DailyMotion", "Pinterest", "Linktree", "Snapchat", "Mastodon"}
        
        results_processed = []
        for f in found[:20]:
            site = f["site"]
            url = f["url"]
            images = []
            text_content = ""
            
            if site in high_value or len(results_processed) < 3:
                # 1. Extract Images
                try:
                    imgs_extracted = extract_images.invoke({"url": url})
                    if isinstance(imgs_extracted, list):
                        images = imgs_extracted
                except Exception:
                    pass
                
                # 2. Extract Textual Data (Bio, names, target-related info)
                try:
                    profile_data = scrape_profile.invoke({"url": url})
                    if isinstance(profile_data, dict):
                        # Keep only the relevant textual bits to avoid huge payloads
                        desc = profile_data.get("description", "") or ""
                        content = profile_data.get("content", "") or ""
                        title = profile_data.get("title", "") or ""
                        
                        # Grab the first 1000 characters of the body to get bios/links
                        snippet = content[:1000] if content else ""
                        text_content = f"Title: {title} | Desc: {desc} | Content: {snippet}"
                except Exception:
                    pass
            
            results_processed.append({
                "site": site,
                "url": url,
                "images": images,
                "profile_text": text_content
            })

        return {
            "username": username,
            "results": results_processed,  # Auto-enriched with images and text
            "count": len(found)
        }

    except Exception as e:
        return {
            "username": username,
            "error": str(e),
            "results": []
        }