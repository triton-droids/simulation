"""Create a local, checksummed T02 handoff; never uploads artifacts."""
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    run=root/'results/post_f1_t02/rate_cost/train'
    target=out/'run'
    target.mkdir()
    for name in ['resolved_config.json','environment_effective_config.json','environment_source.json','robot_source.json','robot_config.json','run_manifest.json','restore_parity.json','train_config.json']:
        shutil.copy2(run/name,target/name)
    shutil.copytree(run/'logs/checkpoints/1003520',target/'logs/checkpoints/1003520')
    for category in ['full_nominal','full_randomized','full_forward_resets']:
        dest=out/'historical_evidence'/category
        dest.mkdir(parents=True)
        for name in ['episodes.csv','summary.json']:
            shutil.copy2(run.parent/category/name,dest/name)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    subprocess.run(['git','archive','--format=zip','--output='+str(out/'source.zip'),'HEAD'],cwd=root,check=True)
    import sys
    versions=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (out/'runtime_versions.txt').write_text(versions)
    hashes={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.rglob('*')) if f.is_file()}
    (out/'manifest.json').write_text(json.dumps({'source_revision':revision,'checkpoint':1003520,'training_seed':11,'scope':'simulation prototype, not three-seed validation','external_dependencies':['pinned MuJoCo Playground and Menagerie caches; see run/*source.json'],'sha256':hashes},indent=2))
    print(out/'manifest.json')

if __name__=='__main__': main()
