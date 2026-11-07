"""Small Field01 panel on the pinned Isaac3.0 Viser visualizer.
Callbacks only enqueue requests; physics changes happen on the simulation thread.
"""
from queue import SimpleQueue
import numpy as np

class FieldPanel:
    def __init__(self,sim,field,identity,seed,preset):
        self.sim=sim;self.field=field;self.events=SimpleQueue();self.mode='Overhead'
        server=sim._visualizers[0]._viewer._server  # Pinned3.0 adapter, checked at startup.
        self.server=server
        with server.gui.add_folder('Soccer field'):
            server.gui.add_markdown(f'**{identity}**\n\nIsaac Lab3.0 / Newton. Seed{seed}, {preset}. Joint hold only; no walking policy.')
            self.pause=server.gui.add_checkbox('Pause field',initial_value=True)
            camera=server.gui.add_dropdown('Camera',options=('Overhead','Orbit','Follow'),initial_value='Overhead')
            camera.on_update(lambda _:self.events.put(('camera',camera.value)))
            reset=server.gui.add_button('Reset robot and ball');reset.on_click(lambda _:self.events.put(('reset',None)))
            self.bx=server.gui.add_slider('Ball X',min=-field.length/2+.3,max=field.length/2-.3,step=.1,initial_value=0.)
            self.by=server.gui.add_slider('Ball Y',min=-field.width/2+.3,max=field.width/2-.3,step=.1,initial_value=0.)
            ball=server.gui.add_button('Place ball');ball.on_click(lambda _:self.events.put(('ball',[self.bx.value,self.by.value])))
            self.rx=server.gui.add_slider('Robot X',min=-field.length/2+.7,max=field.length/2-.7,step=.1,initial_value=-2.)
            self.ry=server.gui.add_slider('Robot Y',min=-field.width/2+.7,max=field.width/2-.7,step=.1,initial_value=0.)
            robot=server.gui.add_button('Place robot');robot.on_click(lambda _:self.events.put(('robot',[self.rx.value,self.ry.value])))
            self.message=server.gui.add_markdown('Paused for inspection. Uncheck Pause field to simulate.')
        server.on_client_connect(lambda client:self.camera())
        self.camera()
    def camera(self,root=None):
        f=self.field
        if self.mode=='Follow' and root is not None:
            target=np.asarray(root);eye=target+np.array([3.,-3.,2.])
        elif self.mode=='Orbit':eye=[f.length,-f.width,9.];target=[0.,0.,0.]
        else:eye=[0.,-.01,f.length*1.4];target=[0.,0.,0.]
        self.sim.set_camera_view(eye,target)
        for client in self.server.get_clients().values():
            client.camera.position=np.asarray(eye,dtype=float)
            client.camera.look_at=np.asarray(target,dtype=float)
            client.camera.up_direction=np.array([0.,1.,0.]) if self.mode=='Overhead' else np.array([0.,0.,1.])
