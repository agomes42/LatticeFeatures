import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os
import sys
from tqdm import tqdm
import argparse

# Add modules directory to path
from modules.Functions import metropolis_algorithm2, action, get_VEV, get_mu

# Check if denoising-diffusion-pytorch is installed
try:
    from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D
except ImportError:
    print("Installing denoising-diffusion-pytorch package...")
    import pip
    pip.main(['install', 'denoising-diffusion-pytorch'])
    from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D


class LatticeDataset(Dataset):
    """Dataset class for lattice field configurations."""
    def __init__(self, data):
        """
        Args:
            data (torch.Tensor): Tensor of shape [num_samples, lattice_size]
        """
        # Store original min and max for later denormalization
        self.data_min = data.min()
        self.data_max = data.max()
        
        # Normalize data to [0,1] range
        normalized_data = (data - self.data_min) / (self.data_max - self.data_min)
        
        # Add channel dimension for compatibility with UNet1D
        self.data = normalized_data.unsqueeze(1)  # Shape becomes [num_samples, 1, lattice_size]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def estimate_autocorrelation_time(lattice_size=32, m2=-1.0, g=1.0, max_iterations=10000):
    """
    Estimate the autocorrelation time for the given parameters
    
    Args:
        lattice_size (int): Size of the lattice
        m2 (float): Mass term squared
        g (float): Coupling constant
        max_iterations (int): Maximum number of iterations to run
        
    Returns:
        int: Estimated autocorrelation time
    """
    print(f"Estimating autocorrelation time for m2={m2}, g={g}...")
    
    # Run a single chain to estimate autocorrelation time
    phi = torch.randn(1, lattice_size)
    accepted = 0
    samples = []
    
    # Run for max_iterations steps
    for i in tqdm(range(max_iterations)):
        # Perform Metropolis step
        phi_new = phi.clone()
        idx = torch.randint(0, lattice_size, (1,))
        phi_new[0, idx] = phi_new[0, idx] + torch.randn(1) * 0.1
        
        # Calculate acceptance probability
        S_old = action(phi, m2=m2, g=g)
        S_new = action(phi_new, m2=m2, g=g)
        delta_S = S_new - S_old
        
        if delta_S <= 0 or torch.rand(1) < torch.exp(-delta_S):
            phi = phi_new
            accepted += 1
        
        # Store the average phi value as our observable
        if i > max_iterations // 10:  # Skip thermalization
            samples.append(torch.mean(phi).item())
    
    # Convert to numpy array
    samples = np.array(samples)
    
    # Calculate autocorrelation function
    mean = np.mean(samples)
    var = np.var(samples)
    norm_samples = samples - mean
    acf = np.correlate(norm_samples, norm_samples, mode='full')
    acf = acf[len(acf)//2:] / (var * len(samples))
    
    # Estimate autocorrelation time (when ACF drops below 1/e)
    try:
        autocorr_time = np.where(acf < 1/np.e)[0][0]
    except:
        # Default if estimation fails
        autocorr_time = 10
        print("Warning: Could not estimate autocorrelation time, using default value of 10")
    
    print(f"Estimated autocorrelation time: {autocorr_time}")
    return autocorr_time


def generate_training_data(lattice_size=32, num_samples=10000, m2=-1.0, g=1.0, num_chains=10):
    """
    Generate training data using the metropolis_algorithm2 function
    
    Args:
        lattice_size (int): Size of the lattice
        num_samples (int): Number of samples to generate
        m2 (float): Mass term squared
        g (float): Coupling constant
        num_chains (int): Number of parallel chains
        
    Returns:
        torch.Tensor: Training data of shape [num_samples, lattice_size]
    """
    # First, estimate the autocorrelation time
    # autocorr_time = estimate_autocorrelation_time(
    #     lattice_size=lattice_size,
    #     m2=m2,
    #     g=g
    # )
    autocorr_time = 50  # changed
    
    # Use at least 2x the autocorrelation time as spacing between samples
    sample_spacing = autocorr_time  # max(2 * autocorr_time, 5)
    
    # Calculate samples per chain
    samples_per_chain = num_samples // num_chains + 1
    
    # Calculate total MC iterations needed
    mc_iterations = samples_per_chain * sample_spacing
    
    # Generate samples
    print(f"Generating {num_samples} training samples using {num_chains} chains...")
    print(f"Using sample spacing of {sample_spacing} based on autocorrelation time")
    
    samples_list = metropolis_algorithm2(
        size=lattice_size,
        mc_iterations=1,  # changed
        m2=m2,
        g=g,
        quiet=False,
        min_samples=samples_per_chain,
        num_chains=num_chains,
        autocorrelation_time=sample_spacing  # Changed parameter name to match the function definition
    )
    
    # Combine samples from all chains
    all_samples = torch.cat([samples[:samples_per_chain] for samples in samples_list if len(samples) >= samples_per_chain], dim=0)
    
    # Trim to desired number of samples
    return all_samples[:num_samples]


def setup_diffusion_model(
    lattice_size=32,
    base_dim=64,
    dim_mults=(1, 2, 4, 8),
    timesteps=1000,
    objective='pred_v'
):
    """
    Setup the UNet1D model and GaussianDiffusion1D process
    """
    # Setup 1D UNet - this defines the neural network architecture
    model = Unet1D(
        dim=base_dim,
        dim_mults=dim_mults,
        channels=1  # Single channel for 1D lattice data
        # Remove resnet_block_groups and attn_dim_head parameters
    )
    
    # Setup Gaussian Diffusion process
    diffusion = GaussianDiffusion1D(
        model,
        seq_length=lattice_size,
        timesteps=timesteps,
        objective=objective  # Can also use 'pred_x0' or 'pred_noise'
    )
    
    return diffusion


def train_diffusion_model(
    diffusion,
    train_data,
    batch_size=32,
    learning_rate=1e-4,
    num_epochs=50,
    save_dir="./results"
):
    """
    Train the diffusion model
    
    Args:
        diffusion (GaussianDiffusion1D): Configured diffusion model
        train_data (torch.Tensor): Training data of shape [num_samples, lattice_size]
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate
        num_epochs (int): Number of training epochs
        save_dir (str): Directory to save checkpoints
        
    Returns:
        tuple: (Trainer1D, LatticeDataset) - Trained diffusion model trainer and dataset
    """
    # Create dataset
    dataset = LatticeDataset(train_data)
    
    # Setup trainer
    trainer = Trainer1D(
        diffusion,
        dataset=dataset,
        train_batch_size=batch_size,
        train_lr=learning_rate,
        train_num_steps=num_epochs,
        save_and_sample_every=400,
        results_folder=save_dir
    )
    
    # Train model
    print("Training diffusion model...")
    trainer.train()
    
    return trainer, dataset


def sample_from_model(diffusion, data_min=None, data_max=None, num_samples=1000, batch_size=32, device='cpu'):
    """
    Sample new lattice configurations from the trained diffusion model
    
    Args:
        diffusion (GaussianDiffusion1D): Trained diffusion model
        data_min (float): Minimum value of the original data for denormalization
        data_max (float): Maximum value of the original data for denormalization
        num_samples (int): Number of samples to generate
        batch_size (int): Batch size for sampling
        device (str): Device to use for sampling
        
    Returns:
        torch.Tensor: Generated samples of shape [num_samples, lattice_size]
    """
    print(f"Generating {num_samples} samples from diffusion model...")
    samples = []
    
    # Move model to device
    diffusion = diffusion.to(device)
    
    # Generate samples in batches
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_size_i = min(batch_size, num_samples - i)
            batch_samples = diffusion.sample(batch_size=batch_size_i)  # [batch_size, 1, lattice_size]
            samples.append(batch_samples.squeeze(1).cpu())  # Remove channel dimension
    
    # Combine all batches
    all_samples = torch.cat(samples, dim=0)
    
    # Denormalize the samples if min and max values are provided
    if data_min is not None and data_max is not None:
        all_samples = all_samples * (data_max - data_min) + data_min
        
    return all_samples


def evaluate_samples(original_samples, generated_samples, m2=-1.0, g=1.0):
    """
    Evaluate the quality of generated samples by comparing observables
    
    Args:
        original_samples (torch.Tensor): Original samples from Metropolis algorithm
        generated_samples (torch.Tensor): Generated samples from diffusion model
        m2 (float): Mass term squared
        g (float): Coupling constant
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    metrics = {}
    
    # Calculate vacuum expectation value (VEV)
    original_vev = get_VEV(original_samples)
    generated_vev = get_VEV(generated_samples)
    metrics['vev_original'] = original_vev.item()
    metrics['vev_generated'] = generated_vev.item()
    
    # Calculate two-point correlation function (mu)
    original_mu = get_mu(original_samples)
    generated_mu = get_mu(generated_samples)
    metrics['mu_original'] = original_mu.item()
    metrics['mu_generated'] = generated_mu.item()
    
    # Calculate action
    original_action = torch.mean(action(original_samples, m2=m2, g=g))
    generated_action = torch.mean(action(generated_samples, m2=m2, g=g))
    metrics['action_original'] = original_action.item()
    metrics['action_generated'] = generated_action.item()
    
    # Print results
    print(f"Original VEV: {metrics['vev_original']:.6f}, Generated VEV: {metrics['vev_generated']:.6f}")
    print(f"Original mu: {metrics['mu_original']:.6f}, Generated mu: {metrics['mu_generated']:.6f}")
    print(f"Original action: {metrics['action_original']:.6f}, Generated action: {metrics['action_generated']:.6f}")
    
    return metrics


def plot_samples(original_samples, generated_samples, num_to_plot=5):
    """
    Plot original and generated samples for visual comparison
    
    Args:
        original_samples (torch.Tensor): Original samples from Metropolis algorithm
        generated_samples (torch.Tensor): Generated samples from diffusion model
        num_to_plot (int): Number of samples to plot
    """
    fig, axs = plt.subplots(num_to_plot, 2, figsize=(10, 2*num_to_plot))
    
    for i in range(num_to_plot):
        # Plot original sample
        axs[i, 0].plot(original_samples[i].numpy())
        axs[i, 0].set_title(f"Original sample {i+1}")
        
        # Plot generated sample
        axs[i, 1].plot(generated_samples[i].numpy())
        axs[i, 1].set_title(f"Generated sample {i+1}")
    
    plt.tight_layout()
    plt.show()


def main(args):
    """Main function to run the diffusion model pipeline"""
    # Set device
    device = 'cpu'  # Force CPU since MPS has issues
    print(f"Using device: {device}")
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Generate or load training data
    if args.load_data is not None and os.path.exists(args.load_data):
        print(f"Loading training data from {args.load_data}")
        train_data = torch.load(args.load_data)
    else:
        train_data = generate_training_data(
            lattice_size=args.lattice_size,
            num_samples=args.num_samples,
            m2=args.m2,
            g=args.g,
            num_chains=args.num_chains
        )
        # Save generated data
        if args.save_data:
            data_path = os.path.join(args.save_dir, "training_data.pt")
            torch.save(train_data, data_path)
            print(f"Saved training data to {data_path}")
    
    # Move data to device
    train_data = train_data.to(device)
    
    # Create dataset for normalization parameters
    dataset = LatticeDataset(train_data)
    data_min = dataset.data_min
    data_max = dataset.data_max
    print(f"Data min: {data_min}, max: {data_max}")
    
    # Setup diffusion model
    diffusion = setup_diffusion_model(
        lattice_size=args.lattice_size,
        base_dim=args.base_dim,
        dim_mults=args.dim_mults,
        timesteps=args.timesteps
    ).to(device)
    
    # Train diffusion model (if not loading pre-trained)
    if not args.load_model:
        trainer, _ = train_diffusion_model(
            diffusion=diffusion,
            train_data=train_data,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            save_dir=args.save_dir
        )
    else:
        # Load pre-trained model
        print(f"Loading pre-trained model from {args.load_model}")
        checkpoint = torch.load(args.load_model, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model_state = checkpoint["model"]
            # If model is itself a dict with state_dict
            if isinstance(model_state, dict) and "state_dict" in model_state:
                model_state = model_state["state_dict"]
            diffusion.load_state_dict(model_state)
        elif isinstance(checkpoint, dict) and "ema" in checkpoint and hasattr(checkpoint["ema"], "state_dict"):
            model_state = checkpoint["ema"].state_dict()
            diffusion.load_state_dict(model_state)
        else:
            try:
                diffusion.load_state_dict(checkpoint)
            except:
                print("Error: Could not load model from checkpoint")
                print("Keys in checkpoint:", checkpoint.keys() if isinstance(checkpoint, dict) else "Not a dictionary")
                return None
    
    # Generate samples
    generated_samples = sample_from_model(
        diffusion=diffusion,
        num_samples=args.eval_samples,
        batch_size=args.batch_size,
        device=device,
        data_min=data_min,
        data_max=data_max
    )
    
    # Ensure samples are on CPU for evaluation
    train_data_cpu = train_data.cpu()
    
    # Evaluate samples
    eval_metrics = evaluate_samples(
        original_samples=train_data_cpu[:args.eval_samples],
        generated_samples=generated_samples,
        m2=args.m2,
        g=args.g
    )
    
    # Plot results
    if args.plot:
        plot_samples(
            original_samples=train_data_cpu[:args.plot_samples],
            generated_samples=generated_samples[:args.plot_samples],
            num_to_plot=args.plot_samples
        )
    
    # Save generated samples
    if args.save_samples:
        samples_path = os.path.join(args.save_dir, "generated_samples.pt")
        torch.save(generated_samples, samples_path)
        print(f"Saved generated samples to {samples_path}")
    
    return eval_metrics


if __name__ == "__main__":
    # Initialize MPS device if available
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        print("MPS device available.")
    else:
        print("MPS device not available. Using CPU.")
        
    parser = argparse.ArgumentParser(description="Train diffusion model for lattice field configurations")
    
    # Data generation parameters
    parser.add_argument("--lattice_size", type=int, default=32, help="Size of the lattice")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of training samples")
    parser.add_argument("--num_chains", type=int, default=10, help="Number of parallel chains for Metropolis")
    parser.add_argument("--m2", type=float, default=-1.0, help="Mass term squared")
    parser.add_argument("--g", type=float, default=1.0, help="Coupling constant")
    parser.add_argument("--load_data", type=str, default=None, help="Path to load training data")
    parser.add_argument("--save_data", action="store_true", help="Whether to save training data")
    
    # Model parameters
    parser.add_argument("--base_dim", type=int, default=64, help="Base dimension of the UNet")
    parser.add_argument("--dim_mults", type=int, nargs="+", default=[1, 2, 4, 8], help="Dimension multipliers")
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--save_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--load_model", type=str, default=None, help="Path to load pre-trained model")
    
    # Evaluation parameters
    parser.add_argument("--eval_samples", type=int, default=1000, help="Number of samples for evaluation")
    parser.add_argument("--save_samples", action="store_true", help="Whether to save generated samples")
    
    # Visualization parameters
    parser.add_argument("--plot", action="store_true", help="Whether to plot samples")
    parser.add_argument("--plot_samples", type=int, default=5, help="Number of samples to plot")
    
    args = parser.parse_args()
    main(args)
