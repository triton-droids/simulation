import mujoco
from mujoco import mjx
import jax

def get_sensor_data(
    model: mujoco.MjModel, data: mjx.Data, sensor_name: str
) -> jax.Array:
  """Gets sensor data given sensor name."""
  sensor_id = model.sensor(sensor_name).id
  sensor_adr = model.sensor_adr[sensor_id]
  sensor_dim = model.sensor_dim[sensor_id]
  return data.sensordata[sensor_adr : sensor_adr + sensor_dim]