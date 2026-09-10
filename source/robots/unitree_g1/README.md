# Unitree G1 model adapter

This package contains no robot meshes or copied Menagerie XML. Both the viewer
and training resolve `unitree_g1/scene_mjx.xml` from the sparse checkout pinned
in `model.py`.

Inspect the exact live mapping with:

```bash
python scripts/inspect_unitree_g1.py --no-fetch-model
```

For offline development, set `UNITREE_G1_MODEL_PATH` to a complete local
`scene_mjx.xml` tree or pass `--model`. Such a run is marked as a local,
unpinned override in its provenance record and must not be conflated with the
standard baseline.
