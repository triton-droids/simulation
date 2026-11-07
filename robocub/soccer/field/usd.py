"""Procedural OpenUSD sources shared by Isaac Lab scenes; no external meshes."""
import math
from pathlib import Path
from .geometry import geometry

def write_usd(field,out):
    from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,Gf,Sdf
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    stage=Usd.Stage.CreateNew(str(out/'field.usda'));root=UsdGeom.Xform.Define(stage,'/Field');stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1.)
    for s in geometry(field):
        cube=UsdGeom.Cube.Define(stage,'/Field/'+s['name'].replace('-', 'neg'));cube.CreateSizeAttr(1.)
        cube.AddTranslateOp().Set(Gf.Vec3d(*s['pos']));cube.AddRotateZOp().Set(math.degrees(s['yaw']));cube.AddScaleOp().Set(Gf.Vec3f(*s['size']));cube.CreateDisplayColorAttr([Gf.Vec3f(*s['color'][:3])])
        material=UsdShade.Material.Define(stage,'/Field/Materials/'+s['name'].replace('-','neg'))
        shader=UsdShade.Shader.Define(stage,str(material.GetPath())+'/Surface')
        shader.CreateIdAttr('UsdPreviewSurface');shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*s['color'][:3]))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
        if s['collision']:
            # Newton hides collision geometry. Keep a separate visible surface.
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            visible=UsdGeom.Cube.Define(stage,str(cube.GetPath())+'_visual');visible.CreateSizeAttr(1.)
            visible.AddTranslateOp().Set(Gf.Vec3d(*s['pos']));visible.AddRotateZOp().Set(math.degrees(s['yaw']));visible.AddScaleOp().Set(Gf.Vec3f(*s['size']))
            visible.CreateDisplayColorAttr([Gf.Vec3f(*s['color'][:3])]);UsdShade.MaterialBindingAPI.Apply(visible.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    stage=Usd.Stage.CreateNew(str(out/'proxy.usda'));root=UsdGeom.Xform.Define(stage,'/Proxy');stage.SetDefaultPrim(root.GetPrim());UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1.)
    def body(name,pos,size,mass):
        b=UsdGeom.Xform.Define(stage,'/Proxy/'+name);b.AddTranslateOp().Set(Gf.Vec3d(*pos));UsdPhysics.RigidBodyAPI.Apply(b.GetPrim());UsdPhysics.MassAPI.Apply(b.GetPrim()).CreateMassAttr(mass)
        c=UsdGeom.Cube.Define(stage,'/Proxy/'+name+'/shape');c.CreateSizeAttr(1.);c.AddScaleOp().Set(Gf.Vec3f(*size));c.CreateDisplayColorAttr([Gf.Vec3f(.9,.45,.08)]);UsdPhysics.CollisionAPI.Apply(c.GetPrim());return b
    base=body('base',(0,0,.4),(.2,.3,.8),4.)
    fix=UsdPhysics.FixedJoint.Define(stage,'/Proxy/fixed');fix.CreateBody1Rel().SetTargets([base.GetPath()]);fix.CreateLocalPos0Attr(Gf.Vec3f(0,0,.4))
    for i,sign in enumerate([-1,1]):
        link=body(f'link{i}',(0,sign*.25,.5),(.08,.08,.4),.6)
        j=UsdPhysics.RevoluteJoint.Define(stage,f'/Proxy/joint{i}');j.CreateBody0Rel().SetTargets([base.GetPath()]);j.CreateBody1Rel().SetTargets([link.GetPath()]);j.CreateAxisAttr('Y');j.CreateLocalPos0Attr(Gf.Vec3f(0,sign*.25,.3));j.CreateLocalPos1Attr(Gf.Vec3f(0,0,.2));j.CreateLowerLimitAttr(-45.);j.CreateUpperLimitAttr(45.)
    stage.GetRootLayer().Save()
    return out/'field.usda',out/'proxy.usda'

