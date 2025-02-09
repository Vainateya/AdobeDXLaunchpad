from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import os

# Step 1: Set up Selenium WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# Step 2: Open Adobe Certification Login Page
login_url = "https://certification.adobe.com/auth/oauth-redirect"
driver.get(login_url)

# Step 3: Manually Log In
input("Log in manually and press Enter when done...")

# Navigate to the Adobe Certification Catalog page
certifications_url = "https://certification.adobe.com/certifications"
driver.get(certifications_url)

# Maximize the window to ensure elements are interactable
driver.maximize_window()

# Wait for the page to load
wait = WebDriverWait(driver, 20)

# Step 4: Click the "Show 60" button
try:
    # Locate all dropdown containers
    dropdowns = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "select2-container")))

    # Ensure there are at least 3 dropdowns
    if len(dropdowns) >= 3:
        third_dropdown = dropdowns[2]
        third_dropdown.click()  # Click the third dropdown

        # Wait for the dropdown options to load
        show_60_option = wait.until(
            EC.presence_of_element_located(
                (By.XPATH, '//li[contains(@class, "select2-result") and .//div[contains(text(), "Show 60")]]')
            )
        )

        # Scroll into view and click "Show 60"
        actions = ActionChains(driver)
        actions.move_to_element(show_60_option).perform()
        show_60_option.click()
        print("Clicked 'Show 60' dropdown.")
        time.sleep(5)  # Allow time for certifications to load
    else:
        print("The third dropdown was not found on the page.")
except Exception as e:
    print(f"Error clicking 'Show 60': {e}")
    driver.quit()
    exit()

# Step 5: Locate and click each "Learn More" button
try:
    # Locate all "Learn More" buttons
    learn_more_buttons = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'learn-more')))

    # Loop through each "Learn More" button
    for index in range(len(learn_more_buttons)):
        try:
            # Re-locate the buttons (as the DOM might update after each click)
            learn_more_buttons = driver.find_elements(By.CLASS_NAME, 'learn-more')

            # Click on the next "Learn More" button
            button = learn_more_buttons[index]
            actions = ActionChains(driver)
            actions.move_to_element(button).perform()  # Scroll into view
            button.click()  # Click the button
            time.sleep(5)  # Wait for the page to load

            # Step 6: Scrape certification details
            soup = BeautifulSoup(driver.page_source, "html.parser")
            cert_title = soup.find("h1").text.strip() if soup.find("h1") else f"certification_{index}"
            cert_filename = f"{cert_title.replace(' ', '_')}.html"
            raw_html = driver.page_source
            
            with open(cert_filename, "w", encoding="utf-8") as f:
                f.write(raw_html)

            print(f"Saved: {cert_filename}")

            # Step 8: Go back to the certifications page
            driver.back()
            time.sleep(5)  # Wait to ensure the page loads again
            try:
                # Locate all dropdown containers
                dropdowns = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "select2-container")))

                # Ensure there are at least 3 dropdowns
                if len(dropdowns) >= 3:
                    third_dropdown = dropdowns[2]
                    third_dropdown.click()  # Click the third dropdown

                    # Wait for the dropdown options to load
                    show_60_option = wait.until(
                        EC.presence_of_element_located(
                            (By.XPATH, '//li[contains(@class, "select2-result") and .//div[contains(text(), "Show 60")]]')
                        )
                    )

                    # Scroll into view and click "Show 60"
                    actions = ActionChains(driver)
                    actions.move_to_element(show_60_option).perform()
                    show_60_option.click()
                    print("Clicked 'Show 60' dropdown.")
                    time.sleep(5)  # Allow time for certifications to load
                else:
                    print("The third dropdown was not found on the page.")
            except Exception as e:
                print(f"Error clicking 'Show 60': {e}")
                driver.quit()
                exit()

        except Exception as e:
            print(f"Error processing certification {index}: {e}")

except Exception as e:
    print(f"Error locating 'Learn More' buttons: {e}")

# Step 9: Close the WebDriver
driver.quit()
