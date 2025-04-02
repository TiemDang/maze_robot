# Maze Robot Navigation - Headless Simulation

This project implements and compares two approaches for teaching a robot to navigate through a maze:
1. Genetic Algorithm (GA) - an evolutionary approach
2. Proximal Policy Optimization (PPO) - a reinforcement learning approach

## Setup

### Required Dependencies

```bash
pip install numpy matplotlib pygame opencv-python torch jupyter ipython
```

### Files Structure

- **Algorithm Implementation:**
  - `class_GA.py` - Genetic Algorithm implementation
  - `ppo_agent.py` - PPO implementation with Actor-Critic network
  - `reward_model.py` - Reward function for RL approach

- **Robot Environment:**
  - `Robot_config.py` - Robot configuration and kinematics
  - `Environment.py` - Environment rendering and visualization
  - `kinematic.py` - Robot kinematics calculations
  - `cost_calculate.py` - Cost/fitness functions (for GA)

- **Simulation Notebooks:**
  - `maze_ga_simulation.ipynb` - GA simulation with video recording
  - `maze_ppo_simulation.ipynb` - PPO simulation with video recording
  - `maze_algorithm_comparison.ipynb` - Comparison of both approaches

- **Utility Scripts:**
  - `save_metrics.py` - Helper functions to save and load metrics
  - `evaluate.py` - Evaluate trained PPO models

## Running Headless Simulations

These notebooks are designed to run in a headless environment (like a remote server) while recording videos that can be viewed in the notebook.

### GA Simulation

1. Open the `maze_ga_simulation.ipynb` notebook
2. Run all cells to execute the GA simulation
3. View the generated video and performance metrics in the notebook
4. The simulation saves:
   - `ga_simulation.mp4` - Video of selected generations
   - `ga_metrics.npz` - Performance metrics data
   - `ga_metrics.png` - Plot of performance metrics

### PPO Simulation

1. Open the `maze_ppo_simulation.ipynb` notebook
2. Run all cells to execute the PPO simulation
3. View the generated video and performance metrics in the notebook
4. The simulation saves:
   - `ppo_simulation.mp4` - Video of selected episodes
   - `ppo_best_model.pt` - Best performing model weights
   - `ppo_final_model.pt` - Final model weights
   - `ppo_metrics.npz` - Performance metrics data
   - `ppo_metrics.png` - Plot of performance metrics

### Algorithm Comparison

After running both simulations:

1. Open the `maze_algorithm_comparison.ipynb` notebook
2. Run all cells to visualize side-by-side comparison
3. The notebook will:
   - Display videos from both approaches side by side
   - Compare performance metrics
   - Provide analysis of tradeoffs between approaches

## Customizing Simulations

- **Adjusting GA parameters:**
  - Modify population size, number of generations, mutation rate in the `run_ga_simulation` function
  
- **Adjusting PPO parameters:**
  - Modify learning rate, discount factor, clipping ratio in the `run_ppo_simulation` function
  
- **Visualization settings:**
  - Adjust `record_interval` to control how many frames are recorded
  - Modify `record_generations`/`record_episodes` to specify which iterations to record

## Performance Tips

- For faster simulations, increase the `record_interval` to skip frames
- For better video quality, decrease the `record_interval` to capture more frames
- Running on a headless server will be faster than running with display

## Troubleshooting

- If you see an error about video codecs, try changing `mp4v` to `XVID` in the `setup_video_writer` function
- If pygame initialization fails, make sure the `SDL_VIDEODRIVER=dummy` environment variable is set correctly
- For memory issues, reduce the number of generations/episodes or increase the recording interval
