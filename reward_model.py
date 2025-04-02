import math
import numpy as np

class RewardModel:
    def __init__(self, target_position=(450, 900), target_orientation=-np.pi/2):
        self.target_position = target_position
        self.target_orientation = target_orientation
        self.prev_distance = None
        self.prev_pos = None
        self.min_distance_achieved = float('inf')
        
    def reset(self, init_pos):
        self.prev_distance = math.sqrt((init_pos[0] - self.target_position[0])**2 + 
                                     (init_pos[1] - self.target_position[1])**2)
        self.prev_pos = init_pos
        self.min_distance_achieved = self.prev_distance
        
    def calculate_reward(self, robot_state):
        """
        Calculate reward based on robot state
        
        robot_state: dict containing:
            - position (x, y)
            - orientation (theta)
            - velocity (x_dot, y_dot)
            - sensor_data (distances from sensors)
            - crash (boolean)
            - time (elapsed time)
        """
        x, y = robot_state['position']
        theta = robot_state['orientation']
        x_dot, y_dot = robot_state['velocity']
        sensor_data = robot_state['sensor_data']
        crash = robot_state['crash']
        time = robot_state['time']
        
        reward = 0
        done = False
        
        # Calculate current distance to goal
        current_distance = math.sqrt((x - self.target_position[0])**2 + (y - self.target_position[1])**2)
        
        # Update minimum distance achieved
        if current_distance < self.min_distance_achieved:
            self.min_distance_achieved = current_distance
        
        # Progress reward (distance-based)
        distance_improvement = self.prev_distance - current_distance
        reward += distance_improvement * 5.0  # Scaled reward for making progress toward goal
        
        # Orientation reward - more reward when facing the goal
        target_angle = math.atan2(self.target_position[1] - y, self.target_position[0] - x)
        angle_diff = abs(self.normalize_angle(target_angle - theta))
        orientation_reward = (1.0 - angle_diff / np.pi) * 0.1
        reward += orientation_reward
        
        # Velocity reward - encourage smooth, consistent motion
        speed = math.sqrt(x_dot**2 + y_dot**2)
        if speed > 0 and speed < 300:  # Reasonable speed range
            reward += 0.1 * (speed / 300)  # Scale with speed, max 0.1
        
        # Smoothness reward - penalize oscillations and jerky movements
        if self.prev_pos:
            dx = x - self.prev_pos[0]
            dy = y - self.prev_pos[1]
            if dx * x_dot < 0 or dy * y_dot < 0:  # Direction changed
                reward -= 0.1
        
        # Crash penalty
        if crash:
            reward -= 10.0
            done = True
        
        # Wall proximity penalty - using sensor data
        min_sensor_distance = min(sensor_data) if sensor_data else float('inf')
        if min_sensor_distance < 30:  # Close to wall
            wall_penalty = ((30 - min_sensor_distance) / 30) * 0.5
            reward -= wall_penalty
        
        # Goal reward
        if current_distance < 30:  # Within 30 pixels of goal
            goal_reward = (30 - current_distance) / 30 * 10.0
            reward += goal_reward
            
            # Bonus for correct orientation at goal
            if current_distance < 15 and abs(self.normalize_angle(theta - self.target_orientation)) < 0.1:
                reward += 20.0
                done = True  # Successfully completed
        
        # Time penalty to encourage faster solutions
        reward -= 0.01  # Small penalty per step
        
        # Timeout
        if time > 30:  # Same as in the original code
            reward -= 5.0
            done = True
        
        # Update previous values for next calculation
        self.prev_distance = current_distance
        self.prev_pos = (x, y)
        
        return reward, done
    
    def normalize_angle(self, angle):
        """Normalize angle to be between -pi and pi"""
        return ((angle + np.pi) % (2 * np.pi)) - np.pi 