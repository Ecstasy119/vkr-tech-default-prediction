"""Export the 5 project notebooks to PDF via nbconvert html + playwright."""
import subprocess
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / 'notebooks'
OUT = ROOT / 'reports' / 'notebook_exports'
OUT.mkdir(parents=True, exist_ok=True)

NOTEBOOKS = [
    '10_russia_load_and_clean',
    '20_russia_eda_and_models',
    '30_china_load_and_clean',
    '40_china_eda_and_models',
    '50_cross_country_pit',
]

# step 1: convert each notebook to HTML (embedded assets)
print('== step 1: nbconvert -> html ==')
for name in NOTEBOOKS:
    cmd = [sys.executable, '-m', 'jupyter', 'nbconvert',
           '--to', 'html', '--embed-images',
           '--output-dir', str(OUT),
           str(NB_DIR / f'{name}.ipynb')]
    print('  ', name)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print('STDERR:', r.stderr[-500:])
        sys.exit(1)

# step 2: render each HTML to PDF via playwright
print('\n== step 2: html -> pdf via Chromium ==')
with sync_playwright() as p:
    browser = p.chromium.launch()
    ctx = browser.new_context()
    for name in NOTEBOOKS:
        html_path = (OUT / f'{name}.html').resolve().as_uri()
        pdf_path = OUT / f'{name}.pdf'
        page = ctx.new_page()
        t0 = time.time()
        page.goto(html_path, wait_until='networkidle')
        page.emulate_media(media='print')
        page.pdf(path=str(pdf_path), format='A4',
                 print_background=True,
                 margin={'top': '15mm', 'bottom': '15mm',
                         'left': '12mm', 'right': '12mm'})
        page.close()
        size_kb = pdf_path.stat().st_size / 1024
        print(f'  {name}.pdf  {size_kb:.0f} KB  ({time.time()-t0:.1f}s)')
    ctx.close()
    browser.close()

# step 3: remove intermediate HTML (keep folder clean)
for name in NOTEBOOKS:
    h = OUT / f'{name}.html'
    if h.exists():
        h.unlink()

print('\ndone. Files in', OUT)
for f in sorted(OUT.glob('*.pdf')):
    print(f'  {f.name}  {f.stat().st_size/1024:.0f} KB')
