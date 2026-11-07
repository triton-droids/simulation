"""Sequential bounded GPU benchmarks; whole-device memory includes other processes."""
import argparse,json,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--counts',nargs='+',type=int,default=[1,16,256]);p.add_argument('--robot',default='g1',choices=['g1','demo']);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=100);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
field_launcher = Path(__file__).with_name("field01")
# Ball dynamics must pass before any throughput number is accepted.
preflight=[sys.executable,str(field_launcher),'smoke','--robot',a.robot,'--num_envs','3','--randomization','disabled','--validate_physics','--steps','10','--run_dir',str(a.output/'physics_preflight')]
with (a.output/'physics_preflight.log').open('w') as log:
    checked=subprocess.run(preflight,stdout=log,stderr=subprocess.STDOUT,timeout=600)
(a.output/'preflight_execution.json').write_text(json.dumps({'command':preflight,'exit_code':checked.returncode},indent=2))
if checked.returncode:sys.exit(checked.returncode)
for n in a.counts:
    dest=a.output/f'n{n}';cmd=[sys.executable,str(field_launcher),'smoke','--robot',a.robot,'--num_envs',str(n),'--steps',str(a.steps),'--randomization','mild','--run_dir',str(dest)]
    samples=[];start=time.monotonic()
    with (a.output/f'n{n}.log').open('w') as log:
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        while child.poll() is None:
            try:
                value=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=5).strip();samples.append({'seconds':time.monotonic()-start,'memory_mib_and_util_percent':value})
            except (OSError,subprocess.SubprocessError):pass
            if time.monotonic()-start>600:child.kill();child.wait();break
            time.sleep(1)
    record={'command':cmd,'exit_code':child.returncode,'seconds':time.monotonic()-start,'memory_scope':'Whole GPU, includes desktop and other processes; not per-process peak','telemetry':samples}
    (a.output/f'n{n}_execution.json').write_text(json.dumps(record,indent=2))
    print(f'N={n}: exit {child.returncode}',flush=True)
    if child.returncode:sys.exit(child.returncode)
