# Legacy remote shell workflows

**The GUI is `App/Scripts/dashboard.py`.** Run that script, or choose **6 RoboCup Dashboard** in PyCharm.

| File | Historical purpose |
|---|---|
| `sweep.sh` | Remote sweep, including Git commit/push and remote checkout preparation |
| `return.sh` | Result downloads; optional remote cleanup depends on local configuration |
| `Outputs/` | Preserved legacy outputs |

These are not the active bounded research launcher. The control panel does not
invoke `sweep.sh`. Keep these paths for historical compatibility; their presence
does not make them recommended entry points. Do not use them for the current campaign.
