from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time

# Setup WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# Open the page
driver.get('https://certification.adobe.com/certifications')

# Maximize the browser window
driver.maximize_window()

# Wait for the page to load
wait = WebDriverWait(driver, 20)

try:
    # Locate all dropdown containers
    dropdowns = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "select2-container")))

    # Ensure there are at least 3 dropdowns
    if len(dropdowns) >= 3:
        # Select the third dropdown
        third_dropdown = dropdowns[2]

        # Click the third dropdown to open it
        third_dropdown.click()

        # Wait for the dropdown options to load
        dropdown_list = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "select2-results")))

        # Locate the "Show 60" option by its visible text
        show_60_option = wait.until(
            EC.presence_of_element_located(
                (By.XPATH, '//li[contains(@class, "select2-result") and .//div[contains(text(), "Show 60")]]')
            )
        )

        # Scroll into view if necessary
        actions = ActionChains(driver)
        actions.move_to_element(show_60_option).perform()

        # Click the "Show 60" option
        show_60_option.click()
        print("Clicked 'Show 60'")

        # Wait for 1 minute after clicking
        time.sleep(60)
    else:
        print("The third dropdown was not found on the page.")

finally:
    # Close the browser
    driver.quit()
