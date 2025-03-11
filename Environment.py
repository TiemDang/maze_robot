import numpy as np
import math
import pygame


class Environment :
    def __init__(self, map_size, path_img):
        # Colors
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)
        self.red = (255, 0, 0)
        self.yellow = (255, 255, 0)
        self.dark_blue = (0, 0, 139)

        # Map size
        self.map_height =  map_size[0]
        self.map_width = map_size[1]

        # Window settings
        pygame.display.set_caption("Robot")
        self.map = pygame.display.set_mode((self.map_width, self.map_height))

        # Load image
        self.path = pygame.image.load(path_img).convert()
        self.path = pygame.transform.scale(self.path, (self.map_width, self.map_height))

        # Text
        self.font = pygame.font.Font('freesansbold.ttf', 30)
        self.textRect = pygame.Rect(self.map_width - 600, self.map_height - 100, 300, 50)

        # Trail
        self.trail_set = []


    def write_info(self, vl, vr, theta):
        text_format = f"Vl = {vl} Vr = {vr} Theta = {int(math.degrees(theta))}"
        text = self.font.render(text_format, True, self.white, self.black)
        self.map.blit(text, self.textRect)


    def sensor_info(self, sensor_data):
        text_format = f"Sensor: {sensor_data}"
        text = self.font.render(text_format, True, self.white, self.black)
        self.map.blit(text, (self.map_width - 700, self.map_height - 50))


    def trail(self, position):
        if len(self.trail_set) > 3000 :
            self.trail_set.pop(0)

        self.trail_set.append(position)

        for i in range(len(self.trail_set) - 1):
            pygame.draw.line(self.map, self.yellow, self.trail_set[i], self.trail_set[i+1], 2)


    def robot_frames(self, position, theta):

        # Draw x_axis and y_axis each time robot rotate a theta deg
        line_lenght = 80
        x, y = position
        """
        # Rotated robot
        x_axis = x + line_lenght * math.cos(-theta + math.pi /2), y + line_lenght * math.sin(-theta + math.pi/2)
        y_axis = x + line_lenght * math.cos(-theta + math.pi), y + line_lenght * math.sin(-theta + math.pi)

        """
        # Origin robot
        x_axis = (x + line_lenght * math.cos(-theta), y + line_lenght * math.sin(-theta))
        y_axis = (x + line_lenght * math.cos(-theta + math.pi/2), y + line_lenght * math.sin(-theta + math.pi/2))
        
        pygame.draw.line(self.map, self.red,(x, y) , x_axis, 3)
        pygame.draw.line(self.map, self.dark_blue,(x, y), y_axis, 3)

    
    def robot_sensor(self, position,theta, points):
        for point in points :
            pygame.draw.line(self.map, self.green, position, point, 2)
            pygame.draw.circle(self.map, self.green, point, 5)

