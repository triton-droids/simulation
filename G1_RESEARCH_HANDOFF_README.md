# G1 Research Handoff Kit, Retrofit Edition

## Current authority — Milestone 1 only (2026-09-27)

The user superseded open-ended three-seed research with **Milestone 1: package,
verify and report the existing simulation controller**, then STOP and pause the
repeating task. Read `research/MILESTONES.md` and `research/milestone_status.json`
first (paths relative to repository root). Milestones 2/3 are deferred and require
new user authorization. All previous results/code/checkpoints remain preserved;
see `research/MILESTONE_2_RESUME.md`. Historical "active" queues and instructions
to continue until three seeds pass below are archival, not current authorization.
No new training or automatic research continuation. R03 was intentionally stopped.


This kit is tailored to the current Triton Droids RoboCup Simulator / PyCharm repository whose top level contains `README.md`, `requirements.txt`, `scripts/`, and `source/`.

## Why this version is different

The repository already contains a pinned Unitree G1 loader and an existing MJX/Brax PPO locomotion stack. The handoff therefore does **not** ask an agent to build a separate top-level `g1_locomotion` project.

Instead, it instructs the agent to:

- preserve `source/locomotion/default_humanoid_legs` as a regression baseline,
- reuse the existing pinned G1 Menagerie model resolver,
- add G1 as a first-class robot/environment under the existing `source/` architecture,
- make cloud logging optional,
- verify and remove hard-coded 12-actuator assumptions from shared seams only where necessary,
- establish a standard G1 velocity baseline before testing HOMIE extensions.

## Install into the repository

Copy the files as follows:

```text
G1_BUILD_AND_RESEARCH_CONTRACT.md       -> repository root
G1_RESEARCH_HANDOFF_README.md           -> repository root
create_codex_research_bundle.bat        -> repository root
create_codex_research_bundle.py         -> scripts/create_codex_research_bundle.py
```

Do not replace `scripts/export_repo_zip.py`. The existing exporter remains useful for ordinary project sharing. The new research exporter is stricter and creates an auditable, secret-scanned handoff bundle.

## Optional local papers

Create:

```text
research/references/papers/
```

and optionally place the papers there:

- HOMIE: https://arxiv.org/pdf/2502.13013
- MuJoCo Playground: https://arxiv.org/pdf/2502.08844

The papers are optional in the kit because the contract tells the executor to use canonical sources when local copies are absent.

## Create a Codex/ChatGPT upload bundle on Windows

Double-click:

```text
create_codex_research_bundle.bat
```

The script writes a timestamped ZIP under:

```text
exports/
```

The bundle excludes Git internals, IDE state, virtual environments, `.cache/` including downloaded Menagerie assets, generated outputs by default, and common credential files. It also scans included text files for common secret patterns and refuses to create the bundle when it detects one.

## Command-line use

From the repository root:

```bash
python scripts/create_codex_research_bundle.py
```

To include generated experiment outputs deliberately:

```bash
python scripts/create_codex_research_bundle.py --include-outputs
```

To pause an IDE run window before closing:

```bash
python scripts/create_codex_research_bundle.py --pause
```

## Recommended instruction after upload

> Read `G1_BUILD_AND_RESEARCH_CONTRACT.md` completely. Audit the repository before editing. Execute the contract gate by gate, beginning with Phase 0. Preserve the existing `default_humanoid_legs` environment as a regression baseline and reuse the repository's pinned Unitree G1 model source rather than creating a second asset path. Continue autonomously through the next safe gate. Do not access remote club infrastructure, push changes, or deploy to hardware without my explicit authorization.

## Important scope note

The current G1 viewer milestone is useful but is not yet a locomotion-training integration. The contract is designed to bridge that exact gap without discarding the working project structure.
