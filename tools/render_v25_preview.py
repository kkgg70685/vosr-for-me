"""Render same-source SVG previews locally; no model or experiment data."""
from pathlib import Path
from playwright.sync_api import sync_playwright

root=Path(__file__).resolve().parents[1]/'docs/architecture_v25'
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True)
    page=browser.new_page(viewport={'width':1560,'height':1000},device_scale_factor=1)
    for name in ('01-model','02-solver','03-evaluation'):
        page.goto((root/(name+'.svg')).as_uri())
        page.screenshot(path=str(root/(name+'.png')))
    browser.close()
print('Three diagram previews rendered with Chromium')
