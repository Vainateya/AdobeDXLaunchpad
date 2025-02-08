from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import time

service = Service(ChromeDriverManager().install())

# Launch Chrome
driver = webdriver.Chrome(service=service)

# Open the login page
login_url = "https://certification.adobe.com/auth/oauth-redirect"  # Replace with the actual login page URL
driver.get(login_url)

# Pause execution to allow manual login
input("Log in manually and press Enter here when done...")

# Navigate to the target page after logging in
target_url = "https://certification.adobe.com/certifications/learn-more"  # Replace with the actual URL you want to scrape
driver.get(target_url)

# Wait for the page to fully load
time.sleep(30)  # Adjust this time if needed

# Extract specific data from the page
elements = driver.find_elements(By.TAG_NAME, "h2")  # Example: Get all <h2> elements

# Print extracted data
for element in elements:
    print(element.text)

# Close the browser when done
driver.quit()
