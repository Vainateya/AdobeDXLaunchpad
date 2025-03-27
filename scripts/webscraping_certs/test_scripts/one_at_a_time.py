from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

# Step 1: Set up Selenium WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# Step 2: Open Adobe Certification Login Page
login_url = "https://certification.adobe.com/auth/oauth-redirect"
driver.get(login_url)

# Step 3: Manually log in
input("Log in manually and press Enter when done...")

# Open the Adobe Certification Catalog page
driver.get("https://certification.adobe.com/certifications")

# Maximize the window to ensure elements are interactable
driver.maximize_window()

# Step 4: Manual scraping loop
try:
    print("\n--- Scraping Started ---")
    print("Navigate to the desired page, then press Enter to scrape.")
    print("Type 'exit' and press Enter to quit.\n")
    
    while True:
        # Wait for user input to trigger scraping
        command = input("Press Enter to scrape or type 'exit' to quit: ").strip()
        
        if command.lower() == "exit":
            print("Exiting scraper.")
            break
        
        # Scrape the current page
        print("Scraping the current page...")
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        # Extract the page title
        page_title = soup.find("h1").text.strip() if soup.find("h1") else "Untitled_Page"
        page_title_sanitized = page_title.replace(" ", "_")
        
        # Extract all text content
        page_text = "\n".join([p.text.strip() for p in soup.find_all(["p", "h2", "h3", "li"]) if p.text.strip()])
        
        # Extract raw HTML for saving
        raw_html = driver.page_source
        
        # Save to a Markdown file
        filename = f"{page_title_sanitized}.md"
        with open(filename, "w", encoding="utf-8") as file:
            file.write(f"# {page_title}\n\n")
            file.write("## Page Content\n\n")
            file.write(page_text + "\n\n")
            file.write("## Raw HTML\n\n")
            file.write(f"```html\n{raw_html}\n```\n")
        
        print(f"Saved content to: {filename}\n")

finally:
    # Close the browser
    driver.quit()
