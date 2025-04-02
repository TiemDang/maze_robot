import pygame
import torch
import numpy as np
import math
import time
from Robot_config import Robot
from Environment import Environment
from ppo_agent import PPO, ActorCritic

# Initialize pygame
pygame.init()

def evaluate_model(model_path, episodes=5, render=True):
    # Load map
    map = pygame.image.load("./maze.jpg")
    map_copy = map.copy()
    
    # Start position & size maze
    start_position = (450, 100)
    end_position = (450, 900, -np.pi/2)
    size = (900, 900)
    
    # Environment setup
    environment = Environment(size, "./maze.jpg")
    
    # Robot setup
    robot = Robot(start_position, "./car.png", 2, 2)
    
    # Setup PPO agent for evaluation
    state_dim = 11  # 8 sensors + x, y, theta
    action_dim = 2  # vx, vtheta
    
    # Create model and load weights
    actor_critic = ActorCritic(state_dim, action_dim)
    actor_critic.load_state_dict(torch.load(model_path))
    actor_critic.eval()  # Set to evaluation mode
    
    # Create PPO agent for evaluation only (not training)
    ppo_agent = PPO(state_dim, action_dim)
    ppo_agent.actor_critic = actor_critic
    
    # Stats collection
    success_count = 0
    distances = []
    times = []
    
    for episode in range(episodes):
        # Reset robot
        robot.reset(start_position)
        
        # Initial state
        robot.update_sensor_data(map_copy, (0, 0, 0))
        state = robot.get_state()
        
        # For visualization
        dt = 0
        last_time = pygame.time.get_ticks()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # Get action from policy (deterministic for evaluation)
            action = ppo_agent.select_action(state, deterministic=True)
            
            # Scale and execute action
            robot.vx = action[0] * 150  # Scale linearly
            robot.vtheta = action[1] * 1.0  # Scale linearly
            
            # Execute action
            robot.move(dt)
            robot.time += dt
            robot.update_sensor_data(map_copy, (0, 0, 0))
            robot.check_crash(map_copy, (0, 0, 0))
            
            # Get new state
            state = robot.get_state()
            
            # Check if episode is done
            distance_to_goal = math.sqrt((robot.x - end_position[0])**2 + (robot.y - end_position[1])**2)
            orientation_diff = abs(((robot.theta - end_position[2] + np.pi) % (2 * np.pi)) - np.pi)
            
            # Check success conditions
            success = distance_to_goal < 30 and orientation_diff < 0.5
            done = robot.crash or robot.time > 30 or success
            
            if done:
                distances.append(distance_to_goal)
                times.append(robot.time)
                if success:
                    success_count += 1
                running = False
            
            # Visualization
            if render:
                robot.draw(environment.map)
                environment.robot_frames([robot.x, robot.y], robot.theta)
                environment.robot_sensor([robot.x, robot.y], robot.points)
                environment.trail((robot.x, robot.y))
                
                # Show additional info
                font = pygame.font.Font('freesansbold.ttf', 20)
                info_text = f"Episode: {episode+1}/{episodes}, Time: {robot.time:.1f}s, Distance: {distance_to_goal:.1f}"
                text = font.render(info_text, True, (255, 255, 255), (0, 0, 0))
                environment.map.blit(text, (10, 10))
                
                dt = (pygame.time.get_ticks() - last_time) / 1000
                last_time = pygame.time.get_ticks()
                pygame.display.update()
                environment.map.blit(map, (0, 0))
                
                # Slow down for better visualization
                time.sleep(0.03)
        
        print(f"Episode {episode+1}: {'Success' if success else 'Failed'}, Distance: {distance_to_goal:.2f}, Time: {robot.time:.2f}s")
    
    # Print summary statistics
    success_rate = success_count / episodes * 100
    avg_distance = sum(distances) / len(distances)
    avg_time = sum(times) / len(times)
    
    print(f"\nEvaluation Results:")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Average Distance to Goal: {avg_distance:.2f}")
    print(f"Average Episode Time: {avg_time:.2f}s")
    
    pygame.quit()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained PPO model for maze navigation')
    parser.add_argument('--model', type=str, default='best_model.pt', help='Path to saved model')
    parser.add_argument('--episodes', type=int, default=5, help='Number of evaluation episodes')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    
    args = parser.parse_args()
    
    evaluate_model(args.model, args.episodes, not args.no_render) 