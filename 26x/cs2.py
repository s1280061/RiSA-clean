from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import os

# 保存先ディレクトリ
out_dir = r"C:\Users\s1280\Desktop\SHRP2rawdata\card"
os.makedirs(out_dir, exist_ok=True)  # フォルダがなければ作成

driver = webdriver.Chrome()  # executable_path は不要
driver.get("file:///C:/Users/s1280/Desktop/SHRP2rawdata/for_GPT/judge_report/index.html")
time.sleep(2)

cards = driver.find_elements(By.CLASS_NAME, "card")
for i, card in enumerate(cards):
    save_path = os.path.join(out_dir, f"card_{i:04}.png")
    card.screenshot(save_path)

driver.quit()
print(f"Saved {len(cards)} cards to {out_dir}")
