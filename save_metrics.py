import numpy as np
import os

def save_ga_metrics(min_fitness, mean_fitness, distances, filename='ga_metrics.npz'):
    """
    Save GA metrics to file for later analysis
    
    Parameters:
    -----------
    min_fitness : array-like
        Minimum fitness values across generations
    mean_fitness : array-like
        Mean fitness values across generations
    distances : array-like
        Best robot's distance to goal across generations
    filename : str
        Name of the file to save metrics to
    """
    np.savez(filename, 
             min_fitness=np.array(min_fitness), 
             mean_fitness=np.array(mean_fitness), 
             distances=np.array(distances))
    print(f"GA metrics saved to {filename}")

def save_ppo_metrics(rewards, distances, filename='ppo_metrics.npz'):
    """
    Save PPO metrics to file for later analysis
    
    Parameters:
    -----------
    rewards : array-like
        Episode rewards during training
    distances : array-like
        Distance to goal at the end of each episode
    filename : str
        Name of the file to save metrics to
    """
    np.savez(filename, 
             rewards=np.array(rewards), 
             distances=np.array(distances))
    print(f"PPO metrics saved to {filename}")

def load_metrics(filename):
    """
    Load saved metrics from file
    
    Parameters:
    -----------
    filename : str
        Name of the file to load metrics from
        
    Returns:
    --------
    dict
        Dictionary containing the metrics
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Metrics file {filename} not found")
    
    return np.load(filename)

def create_comparison_data(ga_metrics_file='ga_metrics.npz', ppo_metrics_file='ppo_metrics.npz', 
                          output_file='comparison_metrics.npz'):
    """
    Create a file with comparison data between GA and PPO
    
    Parameters:
    -----------
    ga_metrics_file : str
        Name of the file with GA metrics
    ppo_metrics_file : str
        Name of the file with PPO metrics
    output_file : str
        Name of the file to save comparison metrics to
    """
    try:
        ga_data = load_metrics(ga_metrics_file)
        ppo_data = load_metrics(ppo_metrics_file)
        
        # Extract distances for comparison
        ga_distances = ga_data['distances']
        ppo_distances = ppo_data['distances']
        
        # Save to a comparison file
        np.savez(output_file,
                ga_distances=ga_distances,
                ppo_distances=ppo_distances,
                ga_iterations=np.arange(1, len(ga_distances) + 1),
                ppo_iterations=np.arange(1, len(ppo_distances) + 1))
        
        print(f"Comparison metrics saved to {output_file}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run both simulations and save metrics first.") 