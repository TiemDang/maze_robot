from Robot_config import Robot
from Environment import Environment
import math
import numpy as np
import pygame

# In pygame 90 mean -90 in kinematics


# Initialize
pygame.init()


# Load map
map = pygame.image.load("/home/venus/venus/maze_robot/mecung.png")
map_copy = map.copy()


# Start position & size maze
start_positon = (440, 0)
size = ([900, 900])
running = True

dt = 0
last_time = pygame.time.get_ticks()
environment = Environment(size, "/home/venus/venus/maze_robot/mecung.png")

# Robot
robot = Robot(start_positon, "/home/venus/venus/maze_robot/car.png", 15, 15)

# Main loop

while running :
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    dt = (pygame.time.get_ticks() - last_time) / 1000
    last_time = pygame.time.get_ticks()


    pygame.display.update()
    
    environment.map.fill(environment.black)
    environment.map.blit(map, (0, 0))
    robot.move(dt)


    robot.draw(environment.map)
    
    environment.trail((robot.x, robot.y))
    environment.robot_frames([robot.x, robot.y], robot.theta)
    environment.write_info(robot.vr, robot.vl, robot.theta)

    pygame.display.update()
    pygame.time.delay(10)

pygame.quit()

