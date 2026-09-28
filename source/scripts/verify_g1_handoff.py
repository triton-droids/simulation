"""Verify packaged sources/checkpoint in an isolated directory using explicit caches."""
import argparse, hashlib, json, os, subprocess, sys, zipfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    package=args.package.resolve();out=args.output_dir.resolve()
    manifest=json.loads((package/'manifest.json').read_text())
    for name,digest in manifest['sha256'].items():
        if hashlib.sha256((package/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Checksum mismatch: '+name)
    out.mkdir(parents=True,exist_ok=False)
    checkout=out/'source'
    with zipfile.ZipFile(package/'source.zip') as z:z.extractall(checkout)
    env=os.environ.copy()
    env['MUJOCO_MENAGERIE_PATH']=str(root/'.cache/mujoco_menagerie')
    env['MUJOCO_PLAYGROUND_PATH']=str(root/'.cache/mujoco_playground')
    subprocess.run([sys.executable,'source/scripts/demo_g1_controller.py','--run-dir',str(package/'run'),'--output-dir',str(out/'smoke'),'--profile','smoke','--video'],cwd=checkout,env=env,check=True)
    (out/'summary.json').write_text(json.dumps({'source_revision':manifest['source_revision'],'checksums_verified':True,'clean_directory_smoke':True,'external_python':sys.executable,'external_caches':[env['MUJOCO_MENAGERIE_PATH'],env['MUJOCO_PLAYGROUND_PATH']]},indent=2))

if __name__=='__main__':main()
