"""Procedural MJCF scene. MuJoCo is an EXPLICIT CPU diagnostic backend."""
import copy,math,xml.etree.ElementTree as ET
from pathlib import Path
from robocub.soccer.field.geometry import geometry

def vec(v):return ' '.join(str(x) for x in v)

def scene_xml(field,robot_info,asset=None):
    if asset:
        root=ET.parse(asset).getroot()
        if root.tag!='mujoco':raise ValueError('MuJoCo field composition currently accepts MJCF only. Convert URDF/USD explicitly with Isaac importer; no silent proxy.')
        root=copy.deepcopy(root)
        compiler=root.find('compiler')
        if compiler is None:compiler=ET.SubElement(root,'compiler')
        for attr in ('meshdir','texturedir'):
            compiler.set(attr,str((Path(asset).resolve().parent/compiler.get(attr,'')).resolve()))
        # Reject unresolved includes rather than silently omit a robot component.
        if root.find('.//include') is not None:raise ValueError('Flatten MJCF includes before field composition')
        for key in list(root.findall('keyframe')):root.remove(key)
        world=root.find('worldbody')
        if world is None:raise ValueError('MJCF missing worldbody')
        if world.find("geom[@name='turf']") is not None:raise ValueError('Robot-only asset required (turf name collision)')
    else:
        root=ET.Element('mujoco',model='field01_DEMO_PROXY_joint_fixture');ET.SubElement(root,'compiler',angle='radian')
        world=ET.SubElement(root,'worldbody')
        body=ET.SubElement(world,'body',name='proxy_root',pos='-2 0 0')
        ET.SubElement(body,'geom',name='proxy_base',type='box',size='.25 .25 .08',pos='0 0 .08',rgba='.9 .45 .08 1')
        ET.SubElement(body,'geom',type='capsule',fromto='0 0 .16 0 0 .8',size='.08',rgba='.9 .45 .08 1')
        for i,sign in enumerate((-1,1)):
            link=ET.SubElement(body,'body',name=f'proxy_link{i}',pos=f'0 {sign*.15} .7')
            ET.SubElement(link,'joint',name=f'proxy_joint{i}',type='hinge',axis='0 1 0',range='-.8 .8',damping='1')
            ET.SubElement(link,'geom',type='capsule',fromto=f'0 0 0 0 {sign*.12} -.4',size='.045',mass='.6',rgba='.9 .45 .08 1')
        act=ET.SubElement(root,'actuator')
        for i in range(2):ET.SubElement(act,'motor',name=f'proxy_motor{i}',joint=f'proxy_joint{i}',ctrllimited='true',ctrlrange='-12 12')
    option=root.find('option')
    if option is None:option=ET.SubElement(root,'option')
    option.set('timestep','.005');option.set('gravity','0 0 -9.81');option.set('integrator','implicitfast')
    vis=root.find('visual')
    if vis is None:vis=ET.SubElement(root,'visual')
    glob=vis.find('global')
    if glob is None:glob=ET.SubElement(vis,'global')
    glob.set('offwidth','1280');glob.set('offheight','800')
    ET.SubElement(world,'light',name='field_light',pos='0 0 12',dir='0 0 -1',diffuse='.8 .8 .8',ambient='.3 .3 .3',directional='true')
    for shape in geometry(field):
        ET.SubElement(world,'geom',name=shape['name'],type='box',pos=vec(shape['pos']),size=vec([v/2 for v in shape['size']]),euler=f"0 0 {shape['yaw']}",rgba=vec(shape['color']),contype='1' if shape['collision'] else '0',conaffinity='1' if shape['collision'] else '0',friction='.8 .005 .0001')
    ball=ET.SubElement(world,'body',name='soccer_ball',pos=f'0 0 {field.ball_radius+.01}')
    ET.SubElement(ball,'freejoint',name='ball_free')
    ET.SubElement(ball,'geom',name='ball',type='sphere',size=str(field.ball_radius),mass=str(field.ball_mass),rgba='.97 .97 .97 1',condim='6',friction='.8 .005 .0001')
    # Visual spots indicate ball rotation without changing contact geometry.
    ET.SubElement(ball,'site',name='ball_spot',type='sphere',size='.018',pos=f'0 0 {field.ball_radius}',rgba='.05 .05 .05 1')
    if robot_info.get('source_scene'):
        contact=ET.parse(robot_info['source_scene']).getroot().find('contact')
        if contact is not None:
            contact=copy.deepcopy(contact)
            for pair in contact:
                if pair.get('geom2')=='floor':pair.set('geom2','turf')
            for side in ('left','right'):
                for k in (1,2,3):ET.SubElement(contact,'pair',geom1=f'{side}_foot{k}_collision',geom2='ball',condim='3',friction='0.8 0.8')
            root.append(contact)
    return ET.tostring(root,encoding='unicode')

