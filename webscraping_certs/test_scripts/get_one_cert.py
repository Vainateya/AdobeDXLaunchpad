from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
import time
import requests
from bs4 import BeautifulSoup

# Step 1: Set up Selenium WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# Step 2: Open Adobe Certification Login Page
login_url = "https://certification.adobe.com/auth/oauth-redirect"
driver.get(login_url)

# Step 3: Manually Log In
input("Log in manually and press Enter when done...")

# Open the Adobe Certification Catalog page
driver.get('https://certification.adobe.com/certifications')

# Maximize the window to ensure elements are interactable
driver.maximize_window()

# Locate all the "Learn More" buttons
learn_more_buttons = driver.find_elements(By.CLASS_NAME, 'learn-more')

# Click the first "Learn More" button if available
if learn_more_buttons:
    first_learn_more_button = learn_more_buttons[0]
    
    # Scroll into view
    actions = ActionChains(driver)
    actions.move_to_element(first_learn_more_button).perform()
    
    # Click the button
    first_learn_more_button.click()
else:
    print("No 'Learn More' buttons found on the page.")

# Optional: Close the browser
driver.quit()
