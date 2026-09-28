# Milestone 1 verification freeze

Zero training. Fixed T02 final1003520 only. Initial GPU evaluation budget7200s.
Eight nominal commands each500steps at reset6000, followed by one continuous
forward500/left-turn500/stand500 episode. Record every failure, RMSE and height;
review all videos. Basic commands must survive and show requested directional
motion. Use original nominal precision gates (stand linear/yaw<=.15, moving
linear<=.35/yaw<=.5), height>.6, no termination. Continuous stopping must settle
to nominal standing precision over its final250steps; whole-segment errors remain
reported, with no claim that this transition check replaces full research gates.
A behavioral failure blocks completion and is reported, not repaired by training.

Package exact checkpoint/configs and historical evidence with SHA256/source archive
and dependency snapshot. Verify every checksum and run a rendered restore smoke
from extracted sources in a clean directory using explicitly provisioned caches
and existing Python environment. This tests relocation, not fresh dependency install.
After queue completion inspect results/videos, finish report and update checklist.
No tracked edits during queue execution. No Milestone2 progression.
