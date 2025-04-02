from Robot_config import Robot
from Environment import Environment
import math
import numpy as np
import pygame
from ppo_agent import PPO
from reward_model import RewardModel
import torch
import time

# In pygame 90 mean -90 in kinematics

# Initialize
pygame.init()

# Load map
map = pygame.image.load("./maze.jpg")
map_copy = map.copy()

# Start position & size maze
start_position = (450, 100)  # origin 450, 100
end_position = (450, 900, -np.pi/2)
size = (900, 900)
running = True

dt = 0
last_time = pygame.time.get_ticks()
environment = Environment(size, "./maze.jpg")

# Robot setup
robot = Robot(start_position, "./car.png", 2, 2)

# Setup PPO agent
# State: [8 sensor readings, x, y, theta]
state_dim = 11  # 8 sensors + x, y, theta
action_dim = 2  # vx, vtheta
ppo_agent = PPO(state_dim=state_dim, action_dim=action_dim, hidden_dim=128, 
               lr=3e-4, gamma=0.99, clip_ratio=0.2, batch_size=32, epochs=5)

# Setup reward model
reward_model = RewardModel(target_position=end_position[:2], target_orientation=end_position[2])

# Training parameters
max_episodes = 1000
max_steps_per_episode = 500
update_frequency = 256  # Update policy after this many state transitions

# Training stats
episode_rewards = []
best_reward = -float('inf')
best_distance = float('inf')

# Training loop
episode = 0

try:
    while running and episode < max_episodes:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Reset environment
        robot.x = start_position[0]
        robot.y = start_position[1]
        robot.theta = np.pi
        robot.vx = 0
        robot.vtheta = 0
        robot.time = 0
        robot.crash = False
        
        # Reset reward model
        reward_model.reset((robot.x, robot.y))
        
        # Initial state (sensor readings + position + orientation)
        robot.update_sensor_data(map_copy, (0, 0, 0))
        state = robot.sensor_data + [float(robot.x), float(robot.y), float(robot.theta)]
        
        episode_reward = 0
        step = 0
        done = False
        
        while not done and step < max_steps_per_episode:
            # Get action from policy
            action, log_prob = ppo_agent.select_action(state)
            
            # Scale action: action[0] is vx, action[1] is vtheta
            robot.vx = action[0] * 150  # Scale to reasonable range
            robot.vtheta = action[1] * 1.0  # Scale to reasonable range
            
            # Execute action
            robot.move(dt)
            robot.time += dt
            robot.update_sensor_data(map_copy, (0, 0, 0))
            robot.check_crash(map_copy, (0, 0, 0))
            
            # Get new state
            next_state = robot.sensor_data + [float(robot.x), float(robot.y), float(robot.theta)]
            
            # Calculate reward and check if done
            robot_state = {
                'position': (robot.x, robot.y),
                'orientation': robot.theta,
                'velocity': (robot.x_dot, robot.y_dot),
                'sensor_data': robot.sensor_data,
                'crash': robot.crash,
                'time': robot.time
            }
            reward, done = reward_model.calculate_reward(robot_state)
            
            # Store transition in PPO memory
            ppo_agent.store_transition(state, action, reward, next_state, done, log_prob)
            
            # Update state and accumulate reward
            state = next_state
            episode_reward += reward
            step += 1
            
            # Visualize (optional, can be disabled for faster training)
            robot.draw(environment.map)
            environment.robot_frames([robot.x, robot.y], robot.theta)
            environment.robot_sensor([robot.x, robot.y], robot.points)
            environment.trail((robot.x, robot.y))
            environment.write_info(robot.vr, robot.vl, robot.theta)
            
            dt = (pygame.time.get_ticks() - last_time) / 1000
            last_time = pygame.time.get_ticks()
            pygame.display.update()
            environment.map.blit(map, (0, 0))
            
            # Uncomment to slow down visualization
            # time.sleep(0.01)
        
        # Update policy
        if (episode + 1) % 5 == 0:  # Update every 5 episodes
            ppo_agent.update()
        
        # Record stats
        episode_rewards.append(episode_reward)
        distance_to_goal = math.sqrt((robot.x - end_position[0])**2 + (robot.y - end_position[1])**2)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(ppo_agent.actor_critic.state_dict(), "best_model.pt")
        
        if distance_to_goal < best_distance:
            best_distance = distance_to_goal
        
        # Print stats
        print(f"Episode: {episode+1}/{max_episodes}, Reward: {episode_reward:.2f}, Distance: {distance_to_goal:.2f}, Best Distance: {best_distance:.2f}")
        
        episode += 1

except KeyboardInterrupt:
    print("Training interrupted by user")
    
finally:
    # Save final model
    torch.save(ppo_agent.actor_critic.state_dict(), "final_model.pt")
    pygame.quit()

# Testing loop (can be run separately with the trained model)
def test_model(model_path):
    # Load the trained model
    ppo_agent.actor_critic.load_state_dict(torch.load(model_path))
    
    # Reset environment
    robot.x = start_position[0]
    robot.y = start_position[1]
    robot.theta = np.pi
    robot.vx = 0
    robot.vtheta = 0
    robot.time = 0
    robot.crash = False
    
    # Test loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get current state
        robot.update_sensor_data(map_copy, (0, 0, 0))
        state = robot.sensor_data + [float(robot.x), float(robot.y), float(robot.theta)]
        
        # Get action from policy (deterministic)
        action = ppo_agent.select_action(state, deterministic=True)
        
        # Scale and execute action
        robot.vx = action[0] * 150
        robot.vtheta = action[1] * 1.0
        
        robot.move(dt)
        robot.check_crash(map_copy, (0, 0, 0))
        
        if robot.crash or robot.time > 30:
            running = False
        
        # Visualization
        robot.draw(environment.map)
        environment.robot_frames([robot.x, robot.y], robot.theta)
        environment.robot_sensor([robot.x, robot.y], robot.points)
        environment.trail((robot.x, robot.y))
        
        dt = (pygame.time.get_ticks() - last_time) / 1000
        last_time = pygame.time.get_ticks()
        pygame.display.update()
        environment.map.blit(map, (0, 0))
        
        robot.time += dt
        
        # Slow down for better visualization
        time.sleep(0.05)

# Uncomment to run test after training
# test_model("best_model.pt")


