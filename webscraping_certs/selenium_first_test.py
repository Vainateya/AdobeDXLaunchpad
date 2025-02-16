from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# Automatically download and use ChromeDriver
service = Service(ChromeDriverManager().install())

# Launch Chrome
driver = webdriver.Chrome(service=service)

# Open a webpage
driver.get("https://google.com")

# Extract page title
print("Page Title:", driver.title)

# Close the browser
driver.quit()
