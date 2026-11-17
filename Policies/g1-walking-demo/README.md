# G1 walking demo

Saved simulation demo; not a three-seed certification.

- [Original run folder](.)
- [Weights](logs/checkpoints/1003520/policy)
- [Resolved configuration](resolved_config.json)
- [Manifest](run_manifest.json)
- [Provenance and SHA256](policy.json)

This folder is a named record, not a copied or promoted policy. Viewing this record
does not change the selected viewer policy. Original experiment paths are preserved.

Experiment ID: `T02`. Intended task: follow forward, sideways and turning velocity commands, including zero-command standing. These are alternative learned policies, not separate movement skills. Experimental replicas are not certified successes.

## What the JSON files mean

These are saved records of this existing policy, not a live training control panel.

| File | Meaning |
|---|---|
| train_config.json | Saved PPO hyperparameters: learning rate, environment count, network sizes, training steps, etc. |
| resolved_config.json | Full resolved training configuration; also read when loading this policy in the viewer. |
| environment_effective_config.json | Effective simulator/environment settings recorded for this run, including rewards and commands. |
| robot_config.json | Robot model structure and joint information. |
| policy.json | Library entry: identity, paths, checksum and validation status. |
| run_manifest.json, environment_source.json, robot_source.json | Run provenance and source information. |
| restore_parity.json | Saved checkpoint restoration verification result. |

For a NEW training recipe, PPO defaults live in robocub/config/agents.py; experiment plans supply explicit overrides. Create a new plan rather than editing historical frozen plans. Changing these saved JSON records does not retrain the weights and may make policy loading inconsistent. Training remains paused.
