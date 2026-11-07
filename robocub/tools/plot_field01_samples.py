"""Plot family selection counts from an existing saved sampling run."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('run_dir',type=Path);a=p.parse_args()
c=json.loads((a.run_dir/'histogram.json').read_text());fig,ax=plt.subplots(figsize=(10,6));ax.barh(list(c),list(c.values()));ax.set_xlabel('Episodes with family enabled');ax.set_title('Field01 deterministic sampler coverage (not physical calibration)');fig.tight_layout();fig.savefig(a.run_dir/'histogram.png',dpi=140)
