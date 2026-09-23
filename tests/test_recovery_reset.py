import unittest
import jax
import jax.numpy as jp
import numpy as np
from source.locomotion.unitree_g1.recovery_reset import sample_recovery_reset, backward_lateral_score

class RecoveryResetTests(unittest.TestCase):
 def test_default_exact_and_selection_preserves_whole_sample(self):
  reset=lambda k:jax.random.uniform(k,(6,),minval=-.5,maxval=.5)
  choose=jax.jit(lambda k:sample_recovery_reset(reset,backward_lateral_score,k,4))
  unchanged=0;changed=0
  for seed in range(40):
   key=jax.random.PRNGKey(seed);base=reset(key)
   np.testing.assert_array_equal(sample_recovery_reset(reset,backward_lateral_score,key,1),base)
   selected=choose(key)
   candidates=[base]+[reset(jax.random.fold_in(key,i)) for i in range(1,4)]
   self.assertTrue(any(np.array_equal(selected,x) for x in candidates))
   self.assertGreaterEqual(float(backward_lateral_score(selected)),float(backward_lateral_score(base)))
   if np.array_equal(selected,base):unchanged+=1
   else:changed+=1
  self.assertGreater(unchanged,0);self.assertGreater(changed,0)
 def test_bounds(self):
  for n in [0,9]:
   with self.assertRaises(ValueError):sample_recovery_reset(lambda k:k,lambda k:0,jax.random.PRNGKey(0),n)
