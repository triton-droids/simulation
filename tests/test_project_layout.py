"""Keep public entry points, docs and IDE buttons consistent after moves."""
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def test_single_public_layout():
    for name in ('App', 'examples', 'START_HERE.md'):
        assert not (ROOT/name).exists()
    for name in ('Scripts/dashboard.py','Scripts/robocub_cli.py',
                 'Scripts/examples/g1_velocity_commands.py','Scripts/legacy/sweep.sh',
                 'Configs/dashboard_plans.json','Policies/catalog.json',
                 'documentation/research/CURRENT.md','documentation/research/CONTRACT.md'):
        assert (ROOT/name).is_file(),name


def test_shared_ide_script_targets_exist():
    for path in (ROOT/'.run').glob('*.run.xml'):
        option = ET.parse(path).find('.//option[@name="SCRIPT_NAME"]')
        target = option.attrib['value'].replace('$PROJECT_DIR$',str(ROOT))
        assert Path(target).is_file(),target


def test_operator_links_resolve():
    docs = [ROOT/'README.md',ROOT/'documentation/README.md']
    docs += list((ROOT/'documentation/guides').rglob('*.md'))
    for path in docs:
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            target=target.split('#')[0]
            if not target or '://' in target:
                continue
            resolved=(path.parent/target).resolve()
            # Generated evidence/weights need provisioning on a fresh checkout.
            if resolved.is_relative_to(ROOT/'Artifacts'):
                continue
            assert resolved.exists(),f'{path}: {target}'


def test_library_and_cli_use_repository_root():
    from Scripts.index_videos import ROOT as library_root
    from robocub.user_cli.cli import T02_RUN
    assert library_root == ROOT
    assert T02_RUN == ROOT/'Policies/T02'
