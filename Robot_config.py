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
        self.image = pygame.transform.scale(self.image,(30, 30))
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
        forward_kinematic = Forward_kinematic((135, -135), (45, -45), l, 0.025, self.theta, 2)
        rotation_inv = forward_kinematic.rotation_matrix()
        j1f_inv = forward_kinematic.J1f_inverse()
        j2 = forward_kinematic.J2()
        phi = np.array([self.vr, self.vl]).reshape(2,1)
        
        
        # Results of  Forward kinematic. x,y and theta
        forward = rotation_inv @ j1f_inv @ j2 @ phi
        x_dot, y_dot, theta_dot = float(forward[0]), float(forward[1]), forward[2]

        self.x = self.x + x_dot * dt * 300 # 300 is a scale between simulate and reality
        self.y = self.y + y_dot * dt * 300
        self.theta = math.radians(self.theta + theta_dot * dt) # pygame use radian
        print(f"{self.x} | {self.y} | {self.theta}")


        # Rotated image
        self.rotated = pygame.transform.rotozoom(self.image, -self.theta, 1)
        self.rect = self.rotated.get_rect(center=(self.x, self.y))


