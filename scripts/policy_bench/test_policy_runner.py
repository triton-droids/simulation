import unittest
import json
import os
import pty
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import numpy as np
from policy_runner import GravityFilter, Policy, parse_imu


class BenchTests(unittest.TestCase):
    def test_units_and_scanner_rejection(self):
        self.assertIsNone(parse_imu('Found I2C device at 0x68','json-si'))
        self.assertIsNone(parse_imu('{"accel_mps2":[0,0,0],"gyro_rad_s":[0,0,0]}','json-si'))
        self.assertIsNone(parse_imu('{"accel_mps2":[0,0,9.8],"gyro_rad_s":[NaN,0,0]}','json-si'))
        acc,gyro,_ = parse_imu('1000,0,0,1,180,0,0','csv-g-dps')
        np.testing.assert_allclose(acc,[0,0,9.80665])
        np.testing.assert_allclose(gyro,[np.pi,0,0])
        acc,gyro,_ = parse_imu('IMU raw | ax=0 ay=0 az=16384 gx=131 gy=0 gz=0','mpu6050-raw')
        np.testing.assert_allclose(acc,[0,0,9.80665])
        np.testing.assert_allclose(gyro,[np.pi/180,0,0])

    def test_gravity_sign_and_gyro_propagation(self):
        filt=GravityFilter()
        np.testing.assert_allclose(filt.update(np.array([0,0,9.80665]),np.zeros(3),0),[0,0,-1])
        # Disable accelerometer correction by using dynamic acceleration.
        result=filt.update(np.array([0,0,15.0]),np.array([1,0,0]),10_000_000)
        self.assertLess(result[1],0)
        self.assertAlmostEqual(np.linalg.norm(result),1)

    def test_policy_reference_and_observation_layout(self):
        from policy_runner import DEFAULT_MODEL
        if not DEFAULT_MODEL.is_file():
            self.skipTest('Download the LFS ONNX export before model-dependent tests')
        p=Policy(DEFAULT_MODEL)
        q=np.arange(10,dtype=np.float32)*0.01
        dq=q+0.5
        previous=q+1
        omega=np.array([0.1,0.2,0.3])
        gravity=np.array([0,0,-1])
        obs=p.observation(7,omega,gravity,q,dq,previous)
        np.testing.assert_array_equal(obs[0,:10],p.reference_q[7])
        np.testing.assert_array_equal(obs[0,10:20],p.reference_dq[7])
        np.testing.assert_allclose(obs[0,20:23],omega)
        np.testing.assert_allclose(obs[0,23:33],q-p.offset)
        np.testing.assert_allclose(obs[0,33:43],dq)
        np.testing.assert_allclose(obs[0,43:53],previous)
        np.testing.assert_allclose(obs[0,53:56],gravity)
        a,target=p.run(obs,7)
        np.testing.assert_allclose(target,p.offset+p.scale*a)
        out=p.session.run(['joint_pos','joint_vel'],{'obs':obs,'time_step':np.array([[7]],np.float32)})
        np.testing.assert_allclose(out[0][0],p.reference_q[7])
        np.testing.assert_allclose(out[1][0],p.reference_dq[7])

    def test_serial_integration_and_stale_stop(self):
        from policy_runner import DEFAULT_MODEL
        if not DEFAULT_MODEL.is_file():
            self.skipTest('Download the LFS ONNX export before the integration test')
        master,slave=pty.openpty()
        port=os.ttyname(slave)
        stop=threading.Event()
        def send():
            began=time.monotonic()
            while not stop.is_set():
                # Stop sending during the run to exercise stale-IMU detection.
                if time.monotonic()-began < 1.5:
                    line=json.dumps({'accel_mps2':[0,0,9.80665],'gyro_rad_s':[0,0,0]})+'\n'
                    os.write(master,line.encode())
                time.sleep(0.005)
        thread=threading.Thread(target=send,daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                out=Path(directory)/'run'
                result=subprocess.run([sys.executable,str(Path(__file__).with_name('policy_runner.py')),'--source','serial','--port',port,'--calibration-seconds','0','--warmup','0','--duration','3','--print-hz','0','--out',str(out)],capture_output=True,text=True,timeout=12)
                summary=json.loads((out/'summary.json').read_text())
                self.assertEqual(result.returncode,1)
                self.assertEqual(summary['status'],'failed')
                self.assertIn('IMU stale',summary['error'])
                self.assertGreater(summary['iterations'],0)
                self.assertEqual(summary['joint_feedback'],'zero_placeholder')
        finally:
            stop.set()
            thread.join(timeout=1)
            os.close(master)
            os.close(slave)


if __name__ == '__main__':
    unittest.main()
