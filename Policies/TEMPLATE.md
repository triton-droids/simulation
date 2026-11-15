# Add a named policy

Create Policies/<name>/ with README.md and policy.json. Include name, validation
status, original run/config/manifest paths, checkpoint step and SHA256 of the weights.
For a packaged demo, keep its complete resolved configuration and checkpoints in
that folder, as T02 does. For an experiment, reference its immutable result directory
instead of duplicating or moving evidence. Add its record to catalog.json and link
it from README.md. Never label a development checkpoint as three-seed validated.

The catalog is descriptive: adding a name does not silently change the selected
viewer policy. Only a compatible, deliberately selected policy should be deployed
in a viewer; arbitrary architectures are not interchangeable.
