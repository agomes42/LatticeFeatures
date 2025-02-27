import torch
import matplotlib.pyplot as plt
import math

# Summary of functions:

# action: Calculate the action of a scalar field on the lattice with periodic boundary conditions.

# get_VEV: Calculate the vacuum expectation value (VEV) of the scalar field.

# get_mu: Calculate the mass of the scalar field around its VEV.

# get_m2_and_g: Calculate the mass term squared (m2) and coupling constant (g) from given parameters.

# metropolis_algorithm: Metropolis algorithm to generate 1D arrays of given size based on a probability 
# distribution function p.

# metropolis_algorithm2: Optimized Metropolis algorithm for multiple chains to generate 1D arrays.

# print_and_plot: Print parameters and plot the generated array with lines at plus and minus VEV.

# point_autocorrelation: Calculate the autocorrelation function of a 1D tensor.

# action_autocorrelation: Calculate the autocorrelation function of the action of a 1D tensor.

# plot_autocorrelation: Plot the point and action autocorrelation functions.

# plot: Plot a graph with optional error bars and log scales.

sig_figs = 3


def action(x: torch.Tensor, m2: float, g: float):
    """
    Calculate the action of a scalar field on the lattice with periodic boundary conditions.
    
    Args:
    x (torch.Tensor): Input array.
    m2 (float): Mass term squared.
    g (float): Coupling constant.
    
    Returns:
    float: Action of the scalar field.
    """
    if m2 < 0:
        constant_term = m2**2 / g**2 / 4
    else:
        constant_term = 0
    
    return torch.sum((x - torch.roll(x, shifts=1, dims=-1))**2 / 2, dim=-1) + torch.sum(g**2 / 4 * x**4 + m2 / 2 * x**2 + constant_term, dim=-1)


def get_VEV(m2: float, g: float):
    """
    Calculate the vacuum expectation value (VEV) of the scalar field.
    
    Args:
    m2 (float): Mass term squared.
    g (float): Coupling constant.
    
    Returns:
    float: Vacuum expectation value.
    """
    if m2 < 0:
        return math.sqrt(-m2) / g
    else:
        return 0


def get_mu(m2: float, g: float):
    """
    Calculate the mass of the scalar field around its VEV.
    
    Args:
    m2 (float): Mass term squared.
    g (float): Coupling constant.
    
    Returns:
    float: Mass around VEV.
    """
    if m2 < 0:
        return math.sqrt(- 2 * m2)
    else:
        return math.sqrt(m2)
    

def get_m2_and_g(abs_m2: float, m2g: float):
    """
    Calculate the mass term squared (m2) and coupling constant (g) from given parameters.
    
    Args:
    abs_m2 (float): Absolute value of mass term squared.
    m2g (float): Dimensionless (having set lattice spacing to 1) parameter related to m2 and g.
    
    Returns:
    tuple: Mass term squared (m2) and coupling constant (g).
    """
    m2 = math.copysign(abs_m2, m2g)
    g = m2 / m2g
    return m2, g


def metropolis_algorithm(size: int = 32, mc_iterations: int = 10000, m2: float = 1.0, g: float = 1.0, quiet: bool = False, autocorrelation_time: int = None, min_samples: int = 1):
    """
    Metropolis algorithm to generate 1D arrays of given size based on a probability distribution function p.
    
    Args:
    size (int): Length of the 1D array.
    mc_iterations (int): Number of Monte Carlo iterations.
    m2 (float): Mass term squared.
    g (float): Coupling constant.
    quiet (bool): If True, suppress output.
    autocorrelation_time (int): Autocorrelation time for sampling.
    min_samples (int): Minimum number of samples to generate.
    
    Returns:
    tuple: Final array and stack of samples.
    """
    v = get_VEV(m2, g)
    mu = get_mu(m2, g)

    #For m2g sufficiently negative, use peaked noise to get over barriers. Not needed for m2g positive, and is singular around m2g = 0
    dist_noise = (m2 / g > -1)

    sqrt_size = torch.tensor(size, dtype=torch.float32).sqrt()
    indices = torch.arange(size, dtype=torch.float32)
    exp_tensor = torch.exp(- mu * torch.abs(indices - size//2))
    mag = 2.5 * 2 / g * math.sqrt(math.sqrt(mu**4 + g**2 * mu / 2) - mu**2)

    # Initialize the array with random values
    # x = torch.randn(size, dtype=torch.float32)
    x = 0.7 * torch.randn(size) / math.sqrt(1 + mu**2 / 2) + v
    x_action = action(x, m2=m2, g=g)
    # print(f"Initial action: {x_action:.{sig_figs}g}")
    
    # keep track of acceptance rate
    accepted = 0
    # random_acceptance = torch.rand(steps)
    samples = []
    time_since_last = 0
    mc_iteration = 0
    while mc_iteration < mc_iterations or len(samples) < min_samples:
        mc_iteration += 1

        # Choose a random index
        # i = torch.randint(0, size, (1,)).item()
        # x_new[i] += torch.randn(1).item()

        # Coefficient tuned to get an acceptance rate around 0.3
        if dist_noise:
            x_new = x + 1.4 * torch.randn(size) / math.sqrt(1 + mu**2 / 2) / sqrt_size
        else:
            shift = torch.randint(0, size, (1,)).item()
            x_new = x + mag * torch.randn(1).item() * torch.roll(exp_tensor, shifts=shift)

        # Calculate the acceptance probability
        x_new_action = action(x_new, m2=m2, g=g)
        acceptance_prob = min(1, torch.exp(-x_new_action + x_action).item())
        
        # Accept or reject the new value
        # if random_acceptance[step].item() < acceptance_prob:
        if torch.rand(1).item() < acceptance_prob:
            x = x_new
            x_action = x_new_action
            accepted += 1
            time_since_last += 1
        
            if autocorrelation_time == None:
                samples.append(x.clone())  # should include even for rejected steps?
            else:
                if time_since_last >= autocorrelation_time:
                    samples.append(x.clone())
                    time_since_last = 0
    
    if not quiet:
        print(f"Acceptance rate: {accepted/mc_iteration:.{sig_figs}g}")
    
    return x, torch.stack(samples)

# ~20x faster for num_chains = 100, a bit slower for num_chains = 1
def metropolis_algorithm2(size: int = 128, mc_iterations: int = 10000, m2: float = 1.0, g: float = 1.0, quiet: bool = False, autocorrelation_time: int = None, min_samples: int = 1, num_chains: int = 1):
    """
    Metropolis algorithm to generate 1D arrays of given size based on a probability distribution function p, optimized for multiple chains.
    
    Args:
    size (int): Length of the 1D array.
    mc_iterations (int): Number of Monte Carlo iterations.
    m2 (float): Mass term squared.
    g (float): Coupling constant.
    quiet (bool): If True, suppress output.
    autocorrelation_time (int): Autocorrelation time for sampling.
    min_samples (int): Minimum number of samples to generate for each chain (may exceed mc_iterations).
    num_chains (int): Number of parallel chains.
    
    Returns:
    list: List of stacks of samples for each chain.
    """
    v = get_VEV(m2, g)
    mu = get_mu(m2, g)

    #For m2g sufficiently negative, use peaked noise to get over barriers. Not needed for m2g positive, and is singular around m2g = 0
    dist_noise = (m2 / g > -1)

    sqrt_size = torch.tensor(size, dtype=torch.float32).sqrt()
    indices = torch.arange(size, dtype=torch.float32).repeat(num_chains, 1)
    mag = 2.5 * 2 / g * math.sqrt(math.sqrt(mu**4 + g**2 * mu / 2) - mu**2)

    # Initialize the array with random values
    x = 0.7 * torch.randn(num_chains, size) / math.sqrt(1 + mu**2 / 2) + v
    x_action = action(x, m2=m2, g=g)
    # print(f"Initial action: {x_action:.{sig_figs}g}")
    
    num_rand = 10000
    random_acceptance = torch.rand(num_rand, num_chains)  # think is ok

    # keep track of acceptance rate
    accepted = 0
    samples_list = [[] for _ in range(num_chains)]
    time_since_last = torch.zeros(num_chains)
    ones = torch.ones(num_chains)
    mc_iteration = 0
    while mc_iteration < mc_iterations or min([len(samples) for samples in samples_list]) < min_samples:
        mc_iteration += 1

        # Coefficient tuned to get an acceptance rate around 0.3
        if dist_noise:
            x_new = x + 1.4 * torch.randn(num_chains, size) / math.sqrt(1 + mu**2 / 2) / sqrt_size
        else:
            pos = torch.randint(0, size, (num_chains,1))
            exp_tensor = torch.exp(- mu * (size//2 - torch.abs(size//2 - torch.abs(indices - pos))))
            x_new = x + mag * torch.randn(1).item() * exp_tensor

        # Calculate the acceptance probability
        x_new_action = action(x_new, m2=m2, g=g)
        acceptance_probs = torch.minimum(torch.exp(-x_new_action + x_action), ones)
        
        # Accept or reject the new value
        accept = random_acceptance[mc_iteration % num_rand] < acceptance_probs
        x = torch.where(accept.unsqueeze(-1), x_new, x)
        x_action = torch.where(accept, x_new_action, x_action)
        accepted += accept.sum().item()
        
        if autocorrelation_time == None:
            for i in torch.where(accept)[0]:
                    samples_list[i].append(x[i].clone())
        else:
            time_since_last += accept
            for i in range(num_chains):
                if time_since_last[i] >= autocorrelation_time:
                    samples_list[i].append(x[i].clone())
                    time_since_last[i] = 0

    acc_rate = accepted/mc_iteration/num_chains
    if acc_rate < 0.2 or acc_rate > 0.45 or not quiet:
        print(f"Acceptance rate: {acc_rate:.{sig_figs}g}")
    
    return [torch.stack(samples) for samples in samples_list] 


def print_and_plot(array, m2, g):
    """
    Print parameters and plot the generated array with lines at plus and minus VEV.
    
    Args:
    array (torch.Tensor): Generated array.
    m2 (float): Mass term squared.
    g (float): Coupling constant.
    """
    v = get_VEV(m2, g)
    mu = get_mu(m2, g)

    print(f'm^2 = {m2}, g = {g}, v = {v:.{sig_figs}g}')
    print(f'elementary mass = {mu:.{sig_figs}g}, rms displacement ~ {1 / math.sqrt(1 + mu**2 / 2):.{sig_figs}g}')
    if m2 < 0:
        s_inst = 4 / 3 / math.sqrt(2) * (-m2)**(3/2) / g**2
        # print(f'S_inst = {s_inst} --> 1/p = {math.exp(s_inst)}')
        print(f'S_inst = {s_inst:.{sig_figs}g}')
    print(f'Final sample action: {action(array, m2=m2, g=g):.{sig_figs}g}')

    # Plot the generated array as well as lines at plus and minus v
    plt.figure(figsize=(5, 3))
    plt.plot(array, label='Field configuation')
    plt.axhline(y=v, color='r', linestyle='--', label='VEV')
    if m2 < 0:
        plt.axhline(y=-v, color='r', linestyle='--')
    plt.legend()
    plt.xlabel('Lattice site')
    plt.ylabel('Field value')
    plt.title(f'Field configuration with m^2 = {m2}, g = {g:.{sig_figs}g}')
    plt.show()


def point_autocorrelation(samples: torch.tensor):
    """
    Calculate the autocorrelation function of a 1D tensor.
    
    Args:
    samples (torch.Tensor): Input tensor of samples.
    
    Returns:
    tuple: Autocorrelation function and estimated autocorrelation time.
    """
    # take first point of each sample
    x = samples[:, 0]

    n = len(x)
    x2_mean = torch.mean(x**2)
    autocorr = torch.zeros(n)
    
    for t in range(n):
        autocorr[t] = torch.sum((x[:n-t]) * (x[t:])) / (n - t) / x2_mean
    
    try:
        time = torch.where(autocorr < torch.exp(torch.tensor(-1.0)))[0][0].item()
    except IndexError:
        print("Point autocorrelation time not found")
        time = n

    return autocorr, time


def action_autocorrelation(samples: torch.tensor, m2: float, g: float, only_time: bool = False):
    """
    Calculate the autocorrelation function of the action of a 1D tensor.
    
    Args:
    samples (torch.Tensor): Input tensor of samples.
    m2 (float): Mass term squared.
    g (float): Coupling constant.
    only_time (bool): If True, return only the estimated autocorrelation time.
    
    Returns:
    tuple: Autocorrelation function and estimated autocorrelation time.
    """
    x = action(samples, m2=m2, g=g)

    n = len(x)
    x_mean = torch.mean(x)
    x_var = torch.var(x)
    autocorr = torch.zeros(n)
    
    for t in range(n):
        autocorr[t] = torch.sum((x[:n-t] - x_mean) * (x[t:] - x_mean)) / (n - t) / x_var
        if only_time and autocorr[t] < torch.exp(torch.tensor(-1.0)):
            return autocorr, t

    try:
        time = torch.where(autocorr < torch.exp(torch.tensor(-1.0)))[0][0].item()
    except IndexError:
        print("Action autocorrelation time not found")
        time = n
    
    return autocorr, time


#plot point and action autocorrelation functions
def plot_autocorrelation(samples: torch.tensor, m2: float, g: float):
    """
    Plot the point and action autocorrelation functions.
    
    Args:
    samples (torch.Tensor): Input tensor of samples.
    m2 (float): Mass term squared.
    g (float): Coupling constant.
    """
    autocorr, time = point_autocorrelation(samples)
    autocorr_action, time_action = action_autocorrelation(samples, m2=m2, g=g)
    print(f"Estimated autocorrelation time: {time} (point), {time_action} (action)")

    factor = 3
    max_time = min(max(factor * time, factor * time_action), len(autocorr))

    plt.figure(figsize=(5, 3))
    plt.plot(autocorr[:max_time], label='Point autocorrelation')
    plt.plot(autocorr_action[:max_time], label='Action autocorrelation')
    plt.axhline(y=torch.exp(torch.tensor(-1.0)), color='r', linestyle='--', label='1/e')
    plt.legend()
    plt.xlabel('Monte Carlo time')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation functions')
    plt.show()


def plot(x, y, xlabel=None, ylabel=None, title=None, y_errors=None, xlog=False, ylog=False, horizontal_line=None):
    """
    Plot a graph with optional error bars and log scales.
    
    Args:
    x (array-like): X-axis data.
    y (array-like): Y-axis data.
    xlabel (str): Label for the X-axis.
    ylabel (str): Label for the Y-axis.
    title (str): Title of the plot.
    y_errors (array-like): Y-axis error bars.
    xlog (bool): If True, use log scale for X-axis.
    ylog (bool): If True, use log scale for Y-axis.
    horizontal_line (float): Y-value for a horizontal line.
    """
    plt.figure(figsize=(5, 3))
    if horizontal_line is not None:
        plt.axhline(y=horizontal_line, color='r', linestyle='--')
    plt.plot(x, y)
    if y_errors is not None:
        plt.fill_between(x, y - y_errors, y + y_errors, alpha=0.2)
    if xlog:
        plt.xscale('log')
    if ylog:
        plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
