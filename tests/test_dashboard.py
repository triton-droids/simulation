import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import pytest
from robocub.user_cli.dashboard import Service, Run, display_state, contained


def test_stale_and_future_heartbeat_not_running():
    now = datetime(2026, 11, 14, 12, tzinfo=timezone.utc)
    for stamp in ['2026-11-14T11:00:00Z', '2026-11-14T13:00:00Z', None]:
        assert display_state({'status':'running','started_at':stamp}, now) != 'running'
    assert display_state({'status':'running','started_at':'2026-11-14T11:59:00Z'}, now) == 'running'


def test_remote_receipt_never_implies_running(tmp_path):
    p = tmp_path/'Artifacts/Results/cluster_research/run/submission.json'
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({'run':'run','status':'submitted','job_id':'123'}))
    assert Service(tmp_path).runs()[0].state == 'unknown — check Slurm'


def test_stop_creates_only_queue_sentinel(tmp_path):
    p = tmp_path/'Artifacts/Results/test/queue/status.json'
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({'status':'running','started_at':datetime.now(timezone.utc).isoformat()}))
    r = Run('test','local','running',p.parent.parent,p)
    Service(tmp_path).stop(r)
    assert (p.parent/'STOP').exists()
    p.write_text('{"status":"needs_review"}')
    with pytest.raises(ValueError): Service(tmp_path).stop(r)


def test_launch_rejects_changed_or_used_plan(tmp_path):
    plan = tmp_path/'plan.json'
    plan.write_text('{"output_dir":"Artifacts/Results/new/queue"}')
    entry = dict(id='new',enabled=True,source='local',plan='plan.json',sha256=hashlib.sha256(plan.read_bytes()).hexdigest())
    catalog = tmp_path/'App/Configs/dashboard_plans.json'
    catalog.parent.mkdir(parents=True)
    catalog.write_text(json.dumps({'plans':[entry]}))
    service = Service(tmp_path)
    assert service.validate_entry(entry)[0] == plan
    plan.write_text(plan.read_text()+' ')
    with pytest.raises(ValueError, match='changed'): service.validate_entry(entry)
    plan.write_text(plan.read_text().rstrip())
    (tmp_path/'Artifacts/Results/new/queue').mkdir(parents=True)
    with pytest.raises(ValueError, match='already exists'): service.validate_entry(entry)


def test_unapproved_and_path_escape_rejected(tmp_path):
    with pytest.raises(ValueError): Service(tmp_path).validate_entry({'source':'local'})
    with pytest.raises(ValueError): contained(tmp_path, '../outside')
    with pytest.raises(ValueError): Service(tmp_path).check('invented_gpu')


def test_completed_run_reports_saved_checkpoint(tmp_path):
    p = tmp_path/'Artifacts/Results/test/queue/status.json'
    p.parent.mkdir(parents=True)
    p.write_text('{"status":"needs_review"}')
    (p.parent.parent/'train/logs/checkpoints/81920').mkdir(parents=True)
    row = Service(tmp_path).runs()[0]
    assert row.step == '81920'
    assert row.state == 'needs review'
