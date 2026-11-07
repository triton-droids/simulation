# Gait diagnostic: straight fine-tune 700

Replayed `outputs/locomotion_straight_finetune_700/humanoid_legs_5_latest.pt`
with `crouch`, `stepping_straight`, forward command 0.35 m/s and zero turn
command. Ten seconds, one robot, seed 42, zero resets. Analysis window:
2.02–10.00 seconds (400 samples at 50 Hz). No policy or reward changes.

## Findings

| Measurement | Left | Right |
|---|---:|---:|
| Sampled positive vertical foot-load share | 70.0% | 30.0% |
| Contact fraction (>1 N net force) | 69.8% | 55.5% |
| Mean vertical foot force over window | 108.1 N | 46.3 N |
| Mean sole-center horizontal speed during contact | 0.028 m/s | 0.124 m/s |
| Mean complete swing duration, excluding events <60 ms | 0.121 s | 0.354 s |
| Complete swings >=60 ms | 20 | 10 |
| Maximum lowest-sole-corner clearance | 40.8 mm | 28.2 mm |

The left foot repeatedly makes a larger, brief lift followed by a smaller
brief lift, rather than one smooth swing. The right foot makes one longer
swing per approximately 0.8-second cycle. Left contact-free advances alternate
roughly 25–28 cm and 4–5 cm; right advances are roughly 31–32 cm. These are
horizontal sole-center distances between sampled liftoff/touchdown boundaries,
not anatomical step lengths. There is one additional 20 ms right-foot contact
dropout with only 0.11 mm clearance; it is retained in raw statistics but
excluded from the duration/count comparison above.

Actual applied actions confirm saturation: the left thigh target stays at
+0.4 rad throughout the window, while the right thigh switches between the
action bounds. The right knee target stays at 0 rad (straight), while the left
knee receives brief -0.8 rad bending targets. The measured left knee reaches
about -25.9 degrees; earlier 10 Hz video-frame telemetry missed the brief peak
and understated its motion. The right knee reaches about -7.3 degrees.

The evidence supports an asymmetric learned gait. The left supports more of
the sampled load, while the right has a longer swing and greater contact-phase
sole movement. It does not establish a mechanical defect or a unique cause.

## Measurement limits and next work

Forces are net foot contact forces sampled at the end of each 20 ms policy
step, not integrated 200 Hz physics impulses. Load shares are approximate;
impact peaks can be undersampled. Contact fractions remain almost unchanged
at 5 N and 10 N thresholds. Sole-center motion during contact can include
rolling/pivoting as well as sliding. Clearance comes from transformed bottom
corners of each actual sole collision box, relative to flat ground z=0.
This is one deterministic replay, not validation across seeds or commands.

Before further long training, investigate why the policy uses saturated
targets and test an opt-in reward/action adjustment that promotes one
continuous swing per leg and reduces stance-foot motion. Compare phase-shifted
left/right timing and motion, rather than demanding identical simultaneous
joint angles or instantaneously equal loads.

Artifacts: [video](gait_diagnostic_700.mp4), [comparison plot](gait_diagnostic_700.png),
[full diagnostics](gait_diagnostic_700.gait.json), [trajectory](gait_diagnostic_700.json).
