"""Render Fig. A3 (figA3_flowchart.html) to figures/output/FigA3_ModelFlowchart.png with headless Chromium.
Requires: pip install playwright && playwright install chromium."""
import os
from playwright.sync_api import sync_playwright
import style as S

src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figA3_flowchart.html')
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 2700, 'height': 1000})
    pg.goto('file://' + src); pg.wait_for_timeout(300)
    pg.screenshot(path=S.paths.fig('FigA3_ModelFlowchart.png'), full_page=True); b.close()
print('->', 'figures/output/FigA3_ModelFlowchart.png')
