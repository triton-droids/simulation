import unittest
import numpy as np
import jax
import jax.numpy as jp
from source.locomotion.unitree_g1.foot_separation import narrow_foot_cost
class FootSeparationTests(unittest.TestCase):
 def test_geometry_and_heading_invariance(self):
  fn=jax.jit(narrow_foot_cost)
  for width,expected in [(.24,0),(.16,0),(.08,.25),(0,1),(-.1,1)]:
   feet=np.array([[0,width/2,0],[0,-width/2,0]])
   self.assertAlmostEqual(float(fn(jp.array(feet),jp.array([1.,0,0,0]))),expected,places=5)
   for yaw in [.7,-2.]:
    c,s=np.cos(yaw),np.sin(yaw);rot=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    self.assertAlmostEqual(float(fn(jp.array(feet@rot.T+3),jp.array([np.cos(yaw/2),0,0,np.sin(yaw/2)]))),expected,places=5)
