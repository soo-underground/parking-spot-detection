"""
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

chrome_options = Options() #
chrome_options.add_argument('--headless')
chrome_options.add_argument("--window-size=1920x1080")
driver = webdriver.Chrome(chrome_options=chrome_options, executable_path=chrome_driver)
driver.get_screenshot_as_file("capture.png")
"""
from selenium import webdriver
from PIL import Image

driver = webdriver.Chrome()
driver.get('https://www.google.co.in')

element = driver.find_element_by_id("lst-ib")

location = element.location
size = element.size

driver.save_screenshot("shot.png")

x = location['x']
y = location['y']
w = size['width']
h = size['height']
width = x + w
height = y + h

im = Image.open('shot.png')
im = im.crop((int(x), int(y), int(width), int(height)))
im.save('image.png')
