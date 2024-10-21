from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from selenium.webdriver.common.by import By
# Create a ChromeOptions object to set the profile
chrome_options = Options()

# Set the path to your Chrome profile's user data
chrome_options.add_argument(r"user-data-dir=C:\Users\praji\AppData\Local\Google\Chrome\User Data")  # Replace with your user data path

# Specify the profile directory (e.g., 'Default', 'Profile 1')
chrome_options.add_argument("profile-directory=Default")  # Or 'Profile 1', 'Profile 2', etc.

# Initialize the Chrome driver with the profile
driver = webdriver.Chrome(options=chrome_options)

# Open any website, in this case, the Colab notebook
driver.get('https://colab.research.google.com/drive/1hQsEnYICFxstDiV0Yzn2hmbvRoRq9P0F?usp=sharing')
time.sleep(30)
connect_button = driver.find_element(By.CSS_SELECTOR,"colab-connect-button")
connect_button.click()
time.sleep(30)
run_button = driver.find_element(By.CSS_SELECTOR,"colab-run-button")
run_button.click()
time.sleep(1000)
# The browser will use the profile where you're already logged in
