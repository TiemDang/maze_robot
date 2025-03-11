import math
import numpy as np
import pygame
from kinematic import Forward_kinematic

# 3m x 3m = 900px x 900px
# 1m =  300px
class Robot :
    def __init__(self, start_postion, image, vl, vr):
        self.x = start_postion[0]
        self.y = start_postion[1]
        self.theta = 90
        self.vr = vr # right wheel
        self.vl = vl # left wheel

        # Graphics
        self.image = pygame.image.load(image)
        self.image = pygame.transform.scale(self.image,(20, 20))
        self.rotated = self.image
        self.rect = self.rotated.get_rect(center=(self.x, self.y))


        # Sensor data
        self.sensor_data = []
        self.points = []

    def draw(self, map):
        map.blit(self.rotated, self.rect)

    
    def move(self, dt):
        l = (math.sqrt(10**2 +  10**2)) / 100 # convert from cm to m
        # Material for Forward kinematic formula
        forward_kinematic = Forward_kinematic((135, -135), (45, -45), l, 0.025, self.theta, 2, (self.vr, self.vl))
        rotation_inv = forward_kinematic.rotation_matrix()
        #j1f_inv = forward_kinematic.J1f_inverse()
        #j2 = forward_kinematic.J2()
        mtrx_2 = forward_kinematic.matrix_2()
        
        
        #phi = np.array([self.vr, self.vl]).reshape(2,1)
        
        
        # Results of  Forward kinematic. x,y and theta
        #forward = rotation_inv @ j1f_inv @ j2 @ phi
        forward = rotation_inv @ mtrx_2
        x_dot, y_dot, theta_dot = float(forward[0]), float(forward[1]), forward[2]

        self.x = self.x + x_dot * dt * 300 # 300 is a scale between simulate and reality
        self.y = self.y + y_dot * dt * 300
        self.theta = self.theta + math.radians(theta_dot* dt)
        #self.theta = self.theta + theta_dot * dt / 300
        # Test 
        print(f"{self.x} | {self.y} | {self.theta} | {math.radians(theta_dot)}")


        # Rotated image
        self.rotated = pygame.transform.rotozoom(self.image, -self.theta, 1)
        self.rect = self.rotated.get_rect(center=(self.x, self.y))

    
    def update_sensor_data(self, track_copy, black_color):
        
        #angles = [self.theta, np.pi/3 + self.theta, 2*np.pi/3 + self.theta, np.pi + self.theta, 4*np.pi/3 + self.theta, 5*np.pi/3 + self.theta]

        angles = [self.theta/180 * np.pi + (-np.pi/2), self.theta/180 * np.pi + (-np.pi), self.theta/180 * np.pi + 3*(-np.pi/2), self.theta/ 180 * np.pi + (-2* np.pi)]

        # position of each sensor [0 = head, 1 = right, 2 = behind, 3 = left]


        edge_points = []
        edge_distances = []


        for angle in angles :
            distance = 0
            edge_x, edge_y = int(self.x), int(self.y)

            while 0 <= edge_x < track_copy.get_width() and 0 <= edge_y < track_copy.get_height() and track_copy.get_at((edge_x, edge_y)) != black_color:
                distance += 1
                edge_x = int(self.x + distance * math.cos(angle))
                edge_y = int(self.y - distance * math.sin(angle))

            edge_points.append((edge_x, edge_y))
            edge_distances.append(distance)


        self.sensor_data = edge_distances
        self.points = edge_points
    
        print(f"h :{self.sensor_data[0]}, r :{self.sensor_data[1]}, bh : {self.sensor_data[2]}, l: {self.sensor_data[3]}")

        """
        # -----Test movement -----------
        if self.sensor_data[0] < 10 :
            self.vl = 15
            self.vr = 10
            self.theta = 180"
        """

            


