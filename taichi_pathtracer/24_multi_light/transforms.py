import math
from dtypes import Vec3f, Mat3f


def rotate(v, yaw, pitch, roll):
    ry = math.radians(yaw)
    rp = math.radians(pitch)
    rr = math.radians(roll)

    sy, cy = math.sin(ry), math.cos(ry)
    sp, cp = math.sin(rp), math.cos(rp)
    sr, cr = math.sin(rr), math.cos(rr)

    x = v.x * (cy * cr + sy * sp * sr) + v.y * (-cp * sr) + v.z * (-sy * cr + cy * sp * sr)
    y = v.x * (cy * sr - sy * sp * cr) + v.y * (cp * cr) + v.z * (-sy * sr - cy * sp * cr)
    z = v.x * (sy * cp) + v.y * (-sp) + v.z * (cy * cp)

    return Vec3f(x, y, z)
