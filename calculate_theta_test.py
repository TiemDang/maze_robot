import math
import numpy as np

# x_axis location ------------
A = (0, 0)
B = (900, 0)
v1 = (B[0]- A[0], B[1] - A[1])
v1_lenght = math.sqrt(v1[0]**2 + v1[1] ** 2)
# random vector

C = (450, 50)
D = (450, 100)
v2 = (D[0] - C[0], D[1] - C[1])
v2_lenght = math.sqrt(v2[0]**2 + v2[1] ** 2)

# degree between two vector

cos_theta =  ( v1[0] * v2[0] ) + (v1[1] * v2[1]) / v1_lenght * v2_lenght


theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
theta_deg = np.degrees(theta)

print(theta_deg)

