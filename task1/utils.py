
import mujoco as mj
import mediapy

def render(xml_path: str, camera_name: str = ""):
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)
    camera_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, camera_name)

    with mj.Renderer(model) as renderer:
        mj.mj_forward(model, data)
        renderer.update_scene(data, camera_id)
        image = renderer.render()
        mediapy.show_image(image)
        
def play(xml_path: str):
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)
    viewer = mj.viewer.launch_passive(model, data)
    viewer.cam.trackbodyid = 0
    viewer.cam.type = mj.viewer.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = 0
    viewer.cam.distance = 2.0
    viewer.cam.lookat[0] = 0.0
    viewer.cam.lookat[1] = 0.0
    viewer.cam.lookat[2] = 0.0