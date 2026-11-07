"""Pinned existing G1 assets, selected by the user for Field01 on2026-10-24."""
import xml.etree.ElementTree as ET
import numpy as np
from robocub.soccer.field.geometry import ROOT
from .importer import inspect_asset

def g1_info(asset_name="g1_mjx.xml"):
    import mujoco
    folder=ROOT/'.cache/mujoco_menagerie/unitree_g1';path=folder/asset_name
    info=inspect_asset(path);m=mujoco.MjModel.from_xml_path(str(folder/('scene_mjx.xml' if asset_name=='g1_mjx.xml' else 'scene.xml')))
    q=m.key_qpos[0] if m.nkey else m.qpos0
    names=[m.joint(int(j)).name for j in m.actuator_trnid[:,0]]
    info.update(name='Unitree G1 29-DOF (pinned Menagerie simulation model)',selection='User requested G1 for now; physical revision not verified',revision='71f066ad0be9cd271f7ed58c030243ef157af9f4',source_scene=str(folder/'scene_mjx.xml'),actuator_gains=m.actuator_gainprm[:,0].tolist(),actuator_damping=(-m.actuator_biasprm[:,2]).tolist(),force_limits=m.jnt_actfrcrange[m.actuator_trnid[:,0]].tolist())
    info['operator_metadata']={'identity':info['name'],'units':'m','license':str(folder/'LICENSE'),'initial_height':float(q[2]),'nominal_joints':{n:float(q[m.jnt_qposadr[j]]) for n,j in zip(names,m.actuator_trnid[:,0])},'kp':float(np.max(m.actuator_gainprm[:,0])),'kd':float(np.max(-m.actuator_biasprm[:,2])),'effort_limit':float(np.max(np.abs(m.jnt_actfrcrange[m.actuator_trnid[:,0]])))}
    if asset_name!='g1_mjx.xml':info.pop('source_scene',None)
    return info
