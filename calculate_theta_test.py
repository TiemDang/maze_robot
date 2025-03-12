import math
import numpy as np

class calculate_degree:
    
    def __init__(self, x_point, y_point, sensor_x, sensor_y) :
        self.x_point = x_point
        self.y_point = y_point
        self.sensor_x = sensor_x
        self.sensor_y = sensor_y

    def cal_degree_vec(self):
        # This is x_axis
        v1 = (self.y_point[0] - self.x_point[0], self.y_point[1] - self.x_point[1])
        v1_lenght = math.sqrt(v1[0] ** 2 + v1[1] ** 2)


        # This is sensor
        v2 = (self.sensor_y[0] - self.sensor_x[0], self.sensor_y[1] - self.sensor_x[1])
        v2_lenght = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        # degree
        cos_theta = (v1[0] * v2[0] + v1[1] * v2[1]) / v1_lenght * v2_lenght
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        theta_deg = np.degrees(theta)

        return theta_deg



"""
init_sss = calculate_degree((0,0),(900,0),(450, 50), (450, 100))
deg = init_sss.cal_degree_vec()
"""



