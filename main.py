from Robot_config import Robot
from Environment import Environment
import math
import numpy as np
import pygame
from class_GA import Genetic_Algo
from ClassNeuralNetwork import NeuralNet
from cost_calculate import calculate_cost
# In pygame 90 mean -90 in kinematics


# Initialize
pygame.init()


# Load map
map = pygame.image.load("/home/venus/venus/maze_robot/maze.jpg")
map_copy = map.copy()


# Start position & size maze
start_positon = (450, 100) # origin 450, 100
end_position = (450, 900, -np.pi/2)
size = (900, 900)
running = True

dt = 0
last_time = pygame.time.get_ticks()
environment = Environment(size, "/home/venus/venus/maze_robot/maze.jpg")

# Robot
#robot = Robot(start_positon, "/home/venus/venus/maze_robot/car.png", 3, 3)

numbers = 30
Robots = []
for i in range(numbers):
    Robots.append(Robot(start_positon, "/home/venus/venus/maze_robot/car.png", 2, 2))


# GA ininit
pop = numbers
GA_init = Genetic_Algo([-5, 5], (-5, 5), 30, 130, 12)
population = GA_init.create_population()


generation = 0
max_generation = 50

# Main loop

while running and generation < max_generation:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    Robot_available = pop 
    decode = GA_init.decode_gen(population)
    for idx, robot in enumerate(Robots) :
        robot.x = start_positon[0]
        robot.y = start_positon[1]
        robot.theta = np.pi
        
        robot.vx = 0
        robot.time = 0
        robot.update_sensor_data(map_copy, (0, 0, 0))

        robot.check_crash(map_copy, (0, 0, 0))
        robot.crash = False

    while Robot_available > 0 :
        for idx, robot in enumerate(Robots) :
            #print(f"Robot {idx}: crash = {robot.crash}, vx = {robot.vx}, vtheta = {robot.vtheta}")
            if robot.crash == False :
                
                # robot.vx = np.random.rand() * 300
                # robot.vtheta = np.random.rand()
                
                decode_individual = decode[idx]
                w = np.reshape(decode_individual[:110],(11, 10))
                v = np.reshape(decode_individual[110:],(10, 2))
                #print(w.shape, v.shape)
                
                
                x = np.array(robot.sensor_data + [float(robot.x), float(robot.y), float(robot.theta)]).reshape(11, 1)

                VVV = NeuralNet(x, w, v).FeedForward()
                #print(VVV.shape)
                robot.vx = (VVV[0][0]) * 2.5
                robot.vtheta = (VVV[1][0]) * 0.03

                ex = end_position[0] - robot.x
                ey = end_position[1] - robot.y
                etheta = end_position[2] - robot.theta
                robot.N = robot.N + 1


                score = calculate_cost(robot.x, robot.y, robot.theta, robot.x_dot, robot.y_dot, robot.sensor_valid, robot.sensor_data)
                robot.cost_function = robot.cost_function + score.euclid_score() + score.sensor_score()
                robot.move(dt)

                robot.time += dt
                robot.check_crash(map_copy, (0,0,0))

                if robot.crash :
                     Robot_available -= 1
                     robot.cost_function = robot.cost_function / robot.N
                     #print(f"Robot {idx}: {robot.cost_function}")
                robot.draw(environment.map)
                robot.update_sensor_data(map_copy, environment.black)
                environment.robot_frames([robot.x, robot.y], robot.theta)
                environment.robot_sensor([robot.x, robot.y], robot.points)
                #environment.trail((robot.x, robot.y))
                #environment.write_info(robot.vr, robot.vl, robot.theta)

        dt = (pygame.time.get_ticks() - last_time) / 1000
        last_time = pygame.time.get_ticks()
        pygame.display.update()
        environment.map.blit(map, (0, 0))

    fitness = []
    for idx, robot in enumerate(Robots) :
        fitness.append(robot.cost_function)
        
    select_population = GA_init.selection(fitness, population)
    cross_population = GA_init.crossover(select_population)
    population = GA_init.mutation(cross_population, 0.2)
    generation = generation + 1

    print(f'Generation: {generation}, j: {np.min(fitness)}, j_mean: {np.mean(fitness)}')
    
    
pygame.quit()


