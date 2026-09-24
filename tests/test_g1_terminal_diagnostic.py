import unittest
from types import SimpleNamespace
import numpy as np
from source.utils.g1_terminal_diagnostic import terminal_signals

class TerminalDiagnosticTests(unittest.TestCase):
 def test_all_causes_and_sensor_addresses(self):
  env=SimpleNamespace(mj_model=SimpleNamespace(sensor_adr=np.array([2,0,3])),
   _right_foot_left_foot_found_sensor=0,_left_foot_right_shin_found_sensor=1,
   _right_foot_left_shin_found_sensor=2,get_gravity=lambda d,f:np.array([0,0,1]))
  data=SimpleNamespace(qpos=np.zeros(7),qvel=np.zeros(6),sensordata=np.zeros(4))
  self.assertFalse(any(bool(v) for k,v in terminal_signals(env,data).items() if k.startswith('terminal/')))
  for address,name in [(2,'foot_foot'),(0,'left_foot_right_shin'),(3,'right_foot_left_shin')]:
   data.sensordata[:]=0;data.sensordata[address]=1
   self.assertTrue(terminal_signals(env,data)['terminal/'+name])
  data.sensordata[:]=0;env.get_gravity=lambda d,f:np.array([0,0,1])
  self.assertFalse(terminal_signals(env,data)['terminal/inverted'])
  env.get_gravity=lambda d,f:np.array([0,0,-.1])
  self.assertTrue(terminal_signals(env,data)['terminal/inverted'])
  data.qvel[0]=np.inf
  self.assertTrue(terminal_signals(env,data)['terminal/invalid'])
