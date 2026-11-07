# Models

`App/Models/T02/` is the canonical local copy of the selected Unitree G1 walking
policy and its exact training configuration.

```text
App/Models/T02/
|-- resolved_config.json
|-- run_manifest.json
`-- logs/checkpoints/1003520/
    |-- policy
    `-- Orbax checkpoint metadata/data
```

The weights file used by visualization is:

```text
App/Models/T02/logs/checkpoints/1003520/policy
```

`App/Models/T02/` is intentionally ignored by Git because it contains generated
binary checkpoint artifacts. Recreating or downloading source code does not
recreate these weights.
