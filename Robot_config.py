import math
import numpy as np
import pygame
from kinematic import Forward_kinematic
from kinematic import Inverse_kinematic
from class_GA import Genetic_Algo

# 3m x 3m = 900px x 900px
# 1m =  300px
class Robot :
    def __init__(self, start_position, image, vl, vr):
        self.x = start_position[0]
        self.y = start_position[1]
        self.theta = -np.pi / 2

        self.vl = vl # left wheel
        self.vr = vr # right wheel
        
        self.vx = 0
        self.vy = 0
        self.vtheta = 0

        self.x_dot = 0
        self.y_dot = 0
        self.theta_dot = 0

        self.sensor_valid = 0
        
        # For RL integration
        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_distance = None
        self.prev_action = None
        
        self.junctions = []
        self.N = 0

        # Graphics
        self.image = pygame.image.load(image)
        self.image = pygame.transform.scale(self.image,(60, 60))
        self.rotated = self.image
        self.rect = self.rotated.get_rect(center=(self.x, self.y))

        # Sensor data
        self.sensor_data = []
        self.points = []

        # Check if car crashing
        self.crash = False
        self.time = 0
        self.cost_function = 0

        # Cost function
        self.frozen_time = 0

    def draw(self, map):
        map.blit(self.rotated, self.rect)

    
    def move(self, dt):
        l = 0.05  # cm 
        # Material for Forward kinematic formula
        kinematic_fw = Forward_kinematic((np.pi /2, -np.pi/2), (0, 0), l, 0.025, self.theta, 2)
        kinematic_inv = Inverse_kinematic((np.pi /2, -np.pi/2), (0, 0), l, 0.025, self.theta, 2)
        
        # Save previous position for reward calculation
        self.prev_x = self.x
        self.prev_y = self.y
        
        # Kinematic variable -----------------------
        rotation_inv = kinematic_fw.inv_rotation_matrix()
        
        j1f = kinematic_inv.J1()
        j1f_inv = kinematic_fw.J1f_inverse()

        
        j2 = kinematic_fw.J2()
        j2_inv = kinematic_inv.J2_inv()

        # Results inverse
        V = j2_inv @ j1f @ np.array([[self.vx], [self.vy], [self.vtheta]])
        self.vr = V[0, 0]
        self.vl = V[1, 0]
        
        
        # Results forward
        forward = rotation_inv @ j1f_inv @ j2 @ np.array([[self.vl], [self.vr]])
        self.x_dot, self.y_dot, self.theta_dot = float(forward[0][0]), float(forward[1][0]), float(forward[2][0])

        
        # Update results
        self.x = self.x + self.x_dot * dt   # 300 is a scale between simulate and reality
        self.y = self.y + self.y_dot * dt 
        self.theta = self.theta + self.theta_dot * dt
        
        # Test (done) 
        #print(f"{self.x} | {self.y} | {self.theta} | {theta_dot}")

        # Rotated image
        theta_deg = math.degrees(self.theta) # Convert radians to deg
        self.rotated = pygame.transform.rotozoom(self.image, theta_deg, 1)
        self.rect = self.rotated.get_rect(center=(self.x, self.y))
    
   
    def update_sensor_data(self, track_copy, black_color):
        
        angles = [self.theta, np.pi/3 + self.theta, 2*np.pi/3 + self.theta, np.pi + self.theta, 4*np.pi/3 + self.theta, 5*np.pi/3 + self.theta, np.pi/2 + self.theta, self.theta - np.pi/2]
        # theta_rad =  ( self.theta / 180 ) * np.pi
        # angles = [-theta_rad + 3 * np.pi / 2, -theta_rad , -theta_rad + np.pi/2]
        
        
        # position of each sensor [0 :left , 1 : head / forward, 2]


        edge_points = []
        edge_distances = []
        width, height = track_copy.get_size()


        for angle in angles :
            distance = 0
            edge_x, edge_y = int(self.x), int(self.y)
            sensor_valid = 0

            while (0 <= edge_x < width and 0 <= edge_y < height and track_copy.get_at((edge_x, edge_y)) != black_color):
                
                edge_x = int(self.x + distance * math.cos(angle))
                edge_y = int(self.y - distance * math.sin(angle))
                distance += 1
                if distance > 110 :
                    sensor_valid = sensor_valid + 1

                if edge_x < 0 or edge_x >= width or edge_y < 0 or edge_y >= height :
                    break

            edge_points.append((edge_x, edge_y))
            edge_distances.append(float(distance))

        #print(self.sensor_data)
        self.sensor_data = edge_distances
        self.points = edge_points
        self.sensor_valid = sensor_valid


    def check_crash(self, track_copy, black_color): # Note fix this code in case x and y is outside range of frame
        edge_x, edge_y = (int(self.x), int(self.y))
        width, height = track_copy.get_size()
        if 0<= edge_x < width and 0 <= edge_y < height :
            if track_copy.get_at((edge_x, edge_y)) == black_color:
                self.crash = True
                self.cost_function = self.cost_function + 10000
        else :
            self.crash = True
            self.cost_function += 10000
        
        if self.time > 30 :
            self.crash = True
            self.cost_function = self.cost_function + 10000
    
    def get_state(self):
        """Return the current state for RL"""
        return self.sensor_data + [float(self.x), float(self.y), float(self.theta)]
    
    def reset(self, start_position=(450, 100), theta=np.pi):
        """Reset the robot to starting position"""
        self.x = start_position[0]
        self.y = start_position[1]
        self.theta = theta
        
        self.vx = 0
        self.vy = 0
        self.vtheta = 0
        
        self.x_dot = 0
        self.y_dot = 0
        self.theta_dot = 0
        
        self.sensor_valid = 0
        self.sensor_data = []
        
        self.crash = False
        self.time = 0
        self.N = 0
        self.cost_function = 0
        
        # Reset rect position
        self.rect = self.rotated.get_rect(center=(self.x, self.y))
        
        return self.get_state()







            


