# Source Ledger

This ledger distinguishes exact ports from adaptations and high-level inspiration. Local implementation must not be described as a reproduction of a source unless the exact source behavior is retained and verified.

| Source | Revision / location used | Precise material consulted | Local use | License / attribution |
|---|---|---|---|---|
| MuJoCo Menagerie, Unitree G1 | Git commit `71f066ad0be9cd271f7ed58c030243ef157af9f4`; `unitree_g1/scene_mjx.xml`, `g1.xml`, `g1_mjx.xml` | Joint, qpos/dof and actuator order; ranges; position gains; `home`/`knees_bent` keyframes; sites, bodies, sensors, collision geoms and explicit pairs | **Exact model source** via resolver; metadata is introspected rather than copied into an asset tree | Unitree G1 directory is BSD-3-Clause, copyright Unitree Robotics. Assets remain in ignored cache and are not vendored. <https://github.com/google-deepmind/mujoco_menagerie/tree/71f066ad0be9cd271f7ed58c030243ef157af9f4/unitree_g1> |
| MuJoCo Playground G1 joystick | Git commit `8a4b4642d8eba8a80ac99ed125cb62c16e1457ad`; `mujoco_playground/_src/locomotion/g1/{base.py,g1_constants.py,joystick.py,randomize.py,xmls/g1_mjx_feetonly.xml,xmls/scene_mjx_feetonly_flat_terrain.xml}`, `_src/{gait.py,mjx_env.py,wrapper.py}`, `config/locomotion_params.py`, `learning/train_jax_ppo.py`, `experimental/sim2sim/{play_g1_joystick.py,onnx/g1_policy.onnx}` | Default configuration; model/scene overlay and expected Menagerie revision `1b86ece576591213e2b666ebf59508454200ca97`; `Joystick._post_init`, `reset`, `step`, `_get_obs`, `_get_termination`, reward/cost helpers and `sample_command`; PD/integration/contact semantics; exact tuned G1 Brax PPO profile; shipped ONNX observation/control loop and policy blob SHA-256 `db2eb258494c1297c43d2b9ffa94cdbde97654c2a44cbab0b40fd4b990752a5b` | **Behavioral reference, pinned runtime adapter, and diagnostic oracle.** The preserved native environment adapts its reward/observation family. The separate thin adapter loads the exact upstream G1, MJX, gait, and Brax-wrapper modules from an ignored verified checkout, while small local code repairs returned observation timing/functional state semantics. It points asset loading at the repository's existing Menagerie resolver; no Playground code or G1 asset tree is copied into tracked source. `source/scripts/evaluate_playground_onnx.py` follows the shipped sim2sim observation/control loop and verifies the blob hash. C05b uses the policy only to falsify model/task defects, not as a local training result. | Apache-2.0, copyright Google LLC. Modified/adapted wrapper files carry a source notice; repository `LICENSE` applies. <https://github.com/google-deepmind/mujoco_playground/tree/8a4b4642d8eba8a80ac99ed125cb62c16e1457ad/mujoco_playground/_src/locomotion/g1> |
| MuJoCo Playground paper | arXiv `2502.08844v1`, 12 Feb 2025; local PDF `research/references/MuJoCo_Playground_2502.08844.pdf` | Sec. II-B joystick task; Sec. III PPO/system design; Appendix B.2 observation/action semantics; Appendix B.3 and Table VI reward family; Appendix E.2 and Table XX G1 PPO hyperparameters, throughput, and training budget | **Method and evaluation reference.** Establishes Playground as the baseline, not a wholesale dependency or claimed reproduction. | Paper citation; no paper text or figures redistributed beyond the supplied local reference. <https://arxiv.org/abs/2502.08844> |
| HOMIE paper | arXiv `2502.13013v2`, 28 Apr 2025; local PDF `research/references/HOMIE_2502.13013.pdf` | Sec. III-A and Fig. 3; Sec. IV-A; Appendix reward/command tables and G1 evaluation table | **Reward/methodology reference only for Gates 0–4.** Phase 4 uses the conventional velocity task; upper-body curriculum, height/knee rewards, and symmetry are explicitly deferred to Phases 5–7. | Paper citation; no code copied. <https://arxiv.org/abs/2502.13013> |
| OpenHOMIE | Not imported for Gates 0–4 | Repository identified by the contract, but no implementation is ported in the standard baseline | **Not used** until separately authorized HOMIE phases require code-level verification | Upstream license must be rechecked before any future adaptation. <https://github.com/InternRobotics/OpenHomie> |
| Brax PPO | Installed `brax==0.14.2`; `brax.training.agents.ppo.train`, networks, and `brax.training.acme.running_statistics` | Existing training integration, asymmetric observation-key interface, checkpoint tuple, wrapper order, and normalized-inference construction | Existing dependency retained for the bounded prototype; no upstream source copied. Commit `f642bc2` makes fixed evaluation use the same running-statistics preprocessor as training. | Apache-2.0. <https://github.com/google/brax> |
| JAX pmap migration guide | Accessed 2026-09-08; current official guide, section “Drop-in replacements for `device_put_sharded` and `device_put_replicated`” | Public `Mesh`, `NamedSharding`, `PartitionSpec`, `device_put`, and tree-map replacement pattern | **Adapted compatibility shim** for helpers removed in JAX 0.11 but still called by Brax 0.14.2. It is installed only by the training entry point and can be deleted when Brax is upgraded. | Apache-2.0 JAX documentation. <https://docs.jax.dev/en/latest/migrate_pmap.html#drop-in-replacements-for-device-put-sharded-and-device-put-replicated> |

## Model compatibility note

C07 additionally inspected the pinned Playground `_src/wrapper.py`
`BraxAutoResetWrapper.step` and installed Brax 0.14.2
`envs/wrappers/training.py:EpisodeWrapper`, `ppo/train.py` and `ppo/losses.py`.
`source/locomotion/unitree_g1/training_wrapper.py` is an original narrow bridge
using Playground's preserve-info hook; it leaves upstream files unchanged and
preserves Brax's terminal/truncation metadata. The action audit mirrors the
installed trainer's normalized-network construction. Both upstream projects
are Apache-2.0. See D-033 for the reproduced failures and validity limits.

The ordinary Menagerie `scene.xml` is valid for MuJoCo viewing but fails the current MJX/Brax path because it contains an unsupported cylinder-mesh collision combination. The pinned `scene_mjx.xml` loads in both MuJoCo and `mjx.put_model`, so the pin does not need to change.

## Evidence-validity note

Fixed evaluations generated before commit `f642bc2` omitted Brax observation
normalization at inference. This invalidates behavior evidence for checkpoints
trained with normalization enabled, from the D-012 runtime-control family
onward. Earlier pilots actually trained with normalization disabled, so identity
inference remains faithful to their unintended profiles. The issue does not
affect the upstream ONNX oracle, which executes its exact shipped graph and raw
103-value sim2sim input contract.
