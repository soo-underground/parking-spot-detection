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

driver = webdriver.Chrome(executable_path=r'D:\curl\chromedriver.exe')
#driver.manage().timeouts().implicitlyWait(5000, TimeUnit.SECONDS);
driver.get('https://www.google.ru')
#driver.get('http://193.169.5.154/mjpg/video.mjpg')

#element = driver.find_element_by_id("lst-ib")

#location = element.location
#size = element.size

driver.save_screenshot("shot.png")

x = 65
y = 144
width = 800
height = 530
#width = x + w
#height = y + h

im = Image.open('shot.png')
im = im.crop((int(x), int(y), int(width), int(height)))
im.save('image.png')
#im.delete('shot.png')
