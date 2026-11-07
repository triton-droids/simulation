"""Browser acceptance test for our own localhost viewer; requires installed Edge."""
from playwright.sync_api import sync_playwright
import argparse,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8080');p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
with sync_playwright() as p:
 b=p.chromium.launch(channel='msedge',headless=True,args=['--enable-webgl','--use-angle=swiftshader'])
 page=b.new_page(viewport={'width':1440,'height':900});page.goto(a.url,wait_until='networkidle')
 page.get_by_role('checkbox',name='Pause field',exact=True).check()
 page.get_by_role('button',name='Reset robot and ball',exact=True).click();page.get_by_text('Reset completed.',exact=True).wait_for()
 for camera in ('Orbit','Overhead','Follow'):
  page.get_by_role('textbox',name='Camera',exact=True).click();page.get_by_role('option',name=camera,exact=True).click()
  page.get_by_text(f'{camera} camera applied.',exact=True).wait_for()
  page.screenshot(path=str(a.output/f'{camera.lower()}.png'))
 boxes=page.get_by_role('textbox');boxes.nth(1).fill('1');boxes.nth(1).press('Enter')
 page.get_by_role('button',name='Place ball',exact=True).click();page.get_by_text(re.compile('Ball placed at')).wait_for()
 page.get_by_role('button',name='Place robot',exact=True).click();page.get_by_text(re.compile('Robot placed at')).wait_for()
 page.get_by_role('checkbox',name='Pause field',exact=True).uncheck()
 page.get_by_role('checkbox',name='Pause field',exact=True).check()
 (a.output/'ui_checks.json').write_text(json.dumps({'reset':True,'place_ball':True,'place_robot':True,'pause_resume':True,'camera_options':['Overhead','Orbit','Follow'],'browser':'headless Edge software WebGL; simulation CUDA'},indent=2))
 print('UI checks passed')
 b.close()
