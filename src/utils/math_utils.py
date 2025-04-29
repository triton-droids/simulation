import numpy as np
import math
import jax.numpy as jp
import jax
import numpy as np
import numpy.typing as npt

ArrayType = jax.Array | npt.NDArray[np.float32]

def quat_inv(quat: ArrayType, order: str = "wxyz") -> ArrayType:
    """Calculates the inverse of a quaternion.

    Args:
        quat (ArrayType): A quaternion represented as an array of four elements.
        order (str, optional): The order of the quaternion components. Either "wxyz" or "xyzw". Defaults to "wxyz".

    Returns:
        ArrayType: The inverse of the input quaternion.
    """
    if order == "xyzw":
        x, y, z, w = quat
    else:
        w, x, y, z = quat

    norm = w**2 + x**2 + y**2 + z**2
    return jp.array([w, -x, -y, -z]) / jp.sqrt(norm)

def quat_mult(q1, q2, order: str = "wxyz"):
    """Multiplies two quaternions and returns the resulting quaternion.

    Args:
        q1 (ArrayType): The first quaternion, represented as an array of four elements.
        q2 (ArrayType): The second quaternion, represented as an array of four elements.
        order (str, optional): The order of quaternion components, either "wxyz" or "xyzw". Defaults to "wxyz".

    Returns:
        ArrayType: The resulting quaternion from the multiplication, in the specified order.
    """
    if order == "wxyz":
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
    else:
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return jp.array([w, x, y, z])

def quat2euler(quat, order: str = "wxyz"):
    """
    Convert a quaternion to Euler angles (roll, pitch, yaw).

    Args:
        quat: Quaternion as [w, x, y, z] or [x, y, z, w].

    Returns:
        Euler angles as [roll, pitch, yaw].
    """
    if order == "xyzw":
        x, y, z, w = quat
    else:
        w, x, y, z = quat

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = jp.arctan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = jp.clip(t2, -1.0, 1.0)
    pitch = jp.arcsin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = jp.arctan2(t3, t4)

    return jp.array([roll, pitch, yaw])

def rotate_vec(vector: ArrayType, quat: ArrayType) -> ArrayType:
    """Rotate a vector using a quaternion.

    Args:
        vector (ArrayType): The vector to be rotated.
        quat (ArrayType): The quaternion representing the rotation.

    Returns:
        ArrayType: The rotated vector.
    """
    v = jp.array([0.0] + list(vector))
    q_inv = quat_inv(quat)
    v_rotated = quat_mult(quat_mult(quat, v), q_inv)
    return v_rotated[1:]

def degrees_to_radians(degrees):
    """Convert angles from degrees to radians."""
    return degrees * (math.pi / 180.0)

def radians_to_degrees(radians):
    """Convert angles from radians to degrees."""
    return radians * (180.0 / math.pi)