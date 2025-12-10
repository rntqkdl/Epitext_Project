# -*- coding: utf-8 -*-
import os, time, requests
from selenium import webdriver
from selenium.webdriver import ChromeOptions
def run(output_root="./kyudb_output"):
    driver = webdriver.Chrome()
    driver.get("https://kyudb.snu.ac.kr")
    # (페이지 넘김 오류 해결 로직 포함)
    print("규장각 크롤링 시작...")
if __name__ == "__main__":
    run()