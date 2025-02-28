import torch
import numpy as np
import matplotlib.pyplot as plt
from modules.Functions import action

def compare_action_distributions(original_samples, generated_samples, m2=-1.0, g=1.0, bins=50, 
                                 title="Action Distribution Comparison", save_path=None, 
                                 density=True, alpha=0.7):
    """
    Compare the action distributions of original and generated samples.
    
    Args:
        original_samples (torch.Tensor): Original samples from Metropolis algorithm
        generated_samples (torch.Tensor): Generated samples from diffusion model
        m2 (float): Mass term squared parameter for action calculation
        g (float): Coupling constant for action calculation
        bins (int): Number of bins for the histogram
        title (str): Plot title
        save_path (str): Path to save the figure, if None, the figure is not saved
        density (bool): Whether to normalize the histogram
        alpha (float): Alpha value for histogram transparency
        
    Returns:
        dict: Dictionary with statistics about both distributions
    """
    # Calculate action for each sample
    original_actions = np.array([action(sample.unsqueeze(0), m2=m2, g=g).item() 
                                for sample in original_samples])
    generated_actions = np.array([action(sample.unsqueeze(0), m2=m2, g=g).item() 
                                 for sample in generated_samples])
    
    # Calculate statistics
    stats = {
        'original_mean': np.mean(original_actions),
        'original_std': np.std(original_actions),
        'original_min': np.min(original_actions),
        'original_max': np.max(original_actions),
        'generated_mean': np.mean(generated_actions),
        'generated_std': np.std(generated_actions),
        'generated_min': np.min(generated_actions),
        'generated_max': np.max(generated_actions),
    }
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    plt.hist(original_actions, bins=bins, alpha=alpha, label=f'Original (μ={stats["original_mean"]:.2f}, σ={stats["original_std"]:.2f})', 
             density=density, color='blue')
    plt.hist(generated_actions, bins=bins, alpha=alpha, label=f'Generated (μ={stats["generated_mean"]:.2f}, σ={stats["generated_std"]:.2f})', 
             density=density, color='red')
    
    # Add vertical lines for means
    plt.axvline(stats['original_mean'], color='blue', linestyle='dashed', linewidth=1)
    plt.axvline(stats['generated_mean'], color='red', linestyle='dashed', linewidth=1)
    
    # Set labels and title
    plt.xlabel('Action (S)')
    plt.ylabel('Probability Density' if density else 'Count')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print statistics
    print(f"Original samples: Mean={stats['original_mean']:.4f}, Std={stats['original_std']:.4f}")
    print(f"Generated samples: Mean={stats['generated_mean']:.4f}, Std={stats['generated_std']:.4f}")
    
    return stats
