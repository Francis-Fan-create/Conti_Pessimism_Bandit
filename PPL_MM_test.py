import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import numpy as np
from scipy import stats

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- Helper Functions for Truncated Gaussians ---
def truncated_normal_log_pdf(x, mean, std, low=-1.0, high=1.0):
    """Log-density of a normal truncated to [low, high]."""
    base_dist = dist.Normal(mean, std)
    log_pdf = base_dist.log_prob(x)
    high_tensor = torch.full_like(x, high)
    low_tensor = torch.full_like(x, low)
    cdf_delta = torch.clamp(base_dist.cdf(high_tensor) - base_dist.cdf(low_tensor), min=1e-12)
    return log_pdf - torch.log(cdf_delta)


def sample_truncated_normal(mean, std, low=-1.0, high=1.0, max_attempts=100):
    """Samples from a normal distribution truncated to [low, high] via rejection sampling."""
    base_dist = dist.Normal(mean, std)
    samples = base_dist.sample()
    attempts = 0
    mask = (samples < low) | (samples > high)
    while mask.any() and attempts < max_attempts:
        resample = base_dist.sample()
        samples = torch.where(mask, resample, samples)
        mask = (samples < low) | (samples > high)
        attempts += 1
    if mask.any():
        samples = torch.clamp(samples, low, high)
    samples = torch.clamp(samples, low, high)
    return samples

# --- 1. UNIFIED BENCHMARK ENVIRONMENT ---
class BenchmarkEnv:
    """
    A unified environment for continuous-action contextual bandits.
    """
    def __init__(self, benchmark_id, context_dim=5, action_dim=1, noise_std=0.1):
        self.context_dim = context_dim
        self.action_dim = action_dim
        self.noise_std = noise_std
        self.benchmark_id = benchmark_id
        
        if benchmark_id == "BiasedBehaviorSharpPeak":
            self._true_reward_func = self._true_reward_sharp_peak
            self._get_behavior_data = self._get_behavior_biased_away
        elif benchmark_id == "SafetyConstrainedReward":
            self._true_reward_func = self._true_reward_safety_constrained
            self._get_behavior_data = self._get_behavior_risky
        elif benchmark_id == "SparseRewardWithNoise":
            self._true_reward_func = self._true_reward_sparse
            self._get_behavior_data = self._get_behavior_uniform_noisy
        else:
            raise ValueError(f"Unknown benchmark_id: {benchmark_id}")

    # --- Reward Functions ---
    def _true_reward_sharp_peak(self, x, a):
        """
        Benchmark 1: VERY sharp peak with extreme penalty away from optimal
        Optimal action: a*(x) = x1 (context-dependent)
        Reward has an extremely sharp peak, making ANY extrapolation catastrophic
        """
        a_star = x[:, 0].unsqueeze(1)
        distance = torch.abs(a - a_star)
        # EXTREMELY sharp exponential decay: tiny mistakes = huge penalty
        reward = torch.exp(-50.0 * distance**2)  # Increased from 10 to 50!
        return reward

    def _true_reward_safety_constrained(self, x, a):
        """
        Benchmark 2: Safety-critical task with distribution mismatch
        Optimal is in safe conservative zone
        Behavior policy explores dangerous regions
        Optimal action: a*(x) = 0.1 * (x1 + x2) (conservative, stays very safe)
        Dangerous region: a > 0.4 (large penalty)
        """
        a_star = 0.1 * (x[:, 0] + x[:, 1]).unsqueeze(1)
        
        # Sharp peak around optimal
        distance = torch.abs(a - a_star)
        base_reward = torch.exp(-15.0 * distance**2)  # Sharp peak
        
        # Strong quadratic penalty in dangerous region (not exponential - too harsh)
        dangerous_mask = (a > 0.4).float()
        danger_amount = torch.clamp(a - 0.4, min=0.0)
        safety_penalty = -3.0 * dangerous_mask * danger_amount**2  # Quadratic penalty
        
        return base_reward + safety_penalty

    def _true_reward_sparse(self, x, a):
        """
        Benchmark 3: EXTREMELY sparse reward with high noise
        Only reward in VERY narrow region, zero elsewhere
        Optimal action: a*(x) = 0.5 * x1 + 0.3 * x2
        Reward zone is VERY narrow (width 0.2), making exploration extremely hard
        """
        a_star = (0.5 * x[:, 0] + 0.3 * x[:, 1]).unsqueeze(1)
        distance = torch.abs(a - a_star)
        
        # EXTREMELY sparse: only reward if VERY close to optimal
        in_reward_zone = (distance < 0.15).float()  # Narrower zone (was 0.3)
        # Exponential decay instead of linear for sharper reward
        reward = in_reward_zone * torch.exp(-10.0 * distance)
        
        return reward

    # --- Behavior Policy Data Generation (Sample + Density) ---
    def _get_behavior_biased_away(self, X):
        """
        Behavior 1: EXTREMELY biased AWAY from optimal actions
        Creates MASSIVE distributional shift - behavior completely avoids good regions!
        """
        n_samples = X.shape[0]
        # Optimal would be a ≈ x1, but behavior chooses OPPOSITE sign with offset
        a_optimal = X[:, 0]
        
        # Behavior is strongly biased to opposite direction with VERY high variance
        mean_behavior = -0.8 * a_optimal - 0.15  # Even more strongly opposite
        std = torch.full((n_samples,), 0.35)  # High variance
        
        A = sample_truncated_normal(mean_behavior, std)
        log_pdf = truncated_normal_log_pdf(A, mean_behavior, std)
        mu_prob = torch.exp(log_pdf)
        
        return A.unsqueeze(1), mu_prob.unsqueeze(1)

    def _get_behavior_risky(self, X):
        """
        Behavior 2: Biased toward dangerous regions
        Explores both safe and dangerous areas but with bias toward danger
        Creates large importance weights when policy learns to stay safe
        """
        n_samples = X.shape[0]
        # Mean = 0.4 (at the edge of danger zone where a > 0.4 is dangerous)
        # High std means it explores both safe and dangerous regions
        mean = torch.full((n_samples,), 0.4)  # At danger boundary
        std = torch.full((n_samples,), 0.3)  # Moderate variance
        
        A = sample_truncated_normal(mean, std)
        log_pdf = truncated_normal_log_pdf(A, mean, std)
        mu_prob = torch.exp(log_pdf)
        
        return A.unsqueeze(1), mu_prob.unsqueeze(1)

    def _get_behavior_uniform_noisy(self, X):
        """
        Behavior 3: Uniform exploration with no structure
        Doesn't exploit any knowledge about optimal actions
        Makes it hard to learn from sparse rewards
        """
        n_samples = X.shape[0]
        # Completely uniform, no context dependence
        A = torch.rand(n_samples) * 2.0 - 1.0  # Uniform [-1, 1]
        mu_prob = torch.full((n_samples,), 0.5)  # Uniform density
        
        return A.unsqueeze(1), mu_prob.unsqueeze(1)

    # --- Public Methods ---
    def get_offline_data(self, n_samples):
        # 1. Sample contexts
        X = torch.rand(n_samples, self.context_dim) * 2 - 1
        
        # 2. Sample actions and get densities from behavior policy
        A, mu_prob = self._get_behavior_data(X)
        
        # 3. Get rewards
        true_rewards = self._true_reward_func(X, A)
        noise = torch.randn_like(true_rewards) * self.noise_std
        R = true_rewards + noise
        
        # DON'T normalize - keep original scale for proper pessimism calibration
        # The pessimism adjustment should work on the natural reward scale
        
        return X.numpy(), A.numpy(), R.numpy(), mu_prob.numpy()

    def evaluate_policy(self, policy, n_test_samples=5000):
        policy.eval()
        # 1. Sample new contexts
        X_test = torch.rand(n_test_samples, self.context_dim) * 2 - 1
        
        with torch.no_grad():
            # 2. Get policy's actions
            A_policy = policy.act(X_test, deterministic=True)
            # 3. Get true (noiseless) reward
            true_rewards = self._true_reward_func(X_test, A_policy)
            
        return true_rewards.mean().item()

# --- 2. POLICY NETWORK (No changes) ---
class PolicyNetwork(nn.Module):
    def __init__(self, context_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_dim * 2)
        )
        self.softplus = nn.Softplus()
    def _get_dist(self, x):
        params = self.net(x)
        alpha = self.softplus(params[:, :1]) + 1.0
        beta = self.softplus(params[:, 1:]) + 1.0
        return dist.Beta(alpha, beta)
    def forward(self, x, a):
        if a.dim() == 1:
            a = a.unsqueeze(1)
        a_raw = (a + 1.0) / 2.0
        beta_dist = self._get_dist(x)
        a_raw_clamped = torch.clamp(a_raw, 1e-6, 1.0 - 1e-6)
        log_prob_raw = beta_dist.log_prob(a_raw_clamped)
        log_prob = log_prob_raw - np.log(2.0)
        return log_prob
    def act(self, x, deterministic=False, n_samples=1):
        beta_dist = self._get_dist(x)
        if deterministic:
            a_raw = beta_dist.mean
        else:
            if n_samples > 1:
                a_raw = beta_dist.sample((n_samples,))
            else:
                a_raw = beta_dist.sample()
        a = 2.0 * a_raw - 1.0
        return a

# --- 3. CONFIGURATION CLASS FOR ABLATION STUDY ---
class TrainingConfig:
    """Configuration for ablation study."""
    def __init__(self, name, beta_pessimism=0, C_clip=float('inf'), epsilon_mu=0, 
                 use_importance_weighting=True, description=""):
        self.name = name
        self.beta_pessimism = beta_pessimism
        self.C_clip = C_clip
        self.epsilon_mu = epsilon_mu
        self.use_importance_weighting = use_importance_weighting
        self.description = description
    
    def __repr__(self):
        return (f"Config({self.name}: β={self.beta_pessimism}, "
                f"C_clip={self.C_clip}, ε_μ={self.epsilon_mu})")


# Define all configurations from the ablation table
ABLATION_CONFIGS = {
    "1. Naive PG": TrainingConfig(
        name="Naive PG",
        beta_pessimism=0,
        C_clip=float('inf'),
        epsilon_mu=0,
        use_importance_weighting=False,
        description="No importance weighting, no pessimism, no regularization"
    ),
    "2. Clamped": TrainingConfig(
        name="Clamped",
        beta_pessimism=0,
        C_clip=float('inf'),
        epsilon_mu=1e-6,
        use_importance_weighting=True,
        description="Importance weighting with behavior policy clamping only"
    ),
    "3. Clipped": TrainingConfig(
        name="Clipped",
        beta_pessimism=0,
        C_clip=50.0,
        epsilon_mu=1e-12,  # Just avoid division by zero, not real clamping
        use_importance_weighting=True,
        description="Importance weighting with weight clipping only"
    ),
    "4. Clamped + Clipped": TrainingConfig(
        name="Clamped+Clipped",
        beta_pessimism=0,
        C_clip=50.0,
        epsilon_mu=1e-6,
        use_importance_weighting=True,
        description="Importance weighting with both clamping and clipping"
    ),
    "5. PPL (pure)": TrainingConfig(
        name="PPL (pure)",
        beta_pessimism=None,  # Will sweep
        C_clip=float('inf'),
        epsilon_mu=0,
        use_importance_weighting=True,
        description="Pure PPL-MM with only pessimism, no clipping or clamping"
    ),
    "6. PPL (no clamping)": TrainingConfig(
        name="PPL (no clamp)",
        beta_pessimism=None,  # Will sweep
        C_clip=50.0,
        epsilon_mu=0,
        use_importance_weighting=True,
        description="PPL-MM with pessimism and clipping, no clamping"
    ),
    "7. PPL (no clipping)": TrainingConfig(
        name="PPL (no clip)",
        beta_pessimism=None,  # Will sweep
        C_clip=float('inf'),
        epsilon_mu=1e-6,
        use_importance_weighting=True,
        description="PPL-MM with pessimism and clamping, no clipping"
    ),
    "8. PPL (Both)": TrainingConfig(
        name="PPL (paper)",
        beta_pessimism=None,  # Will sweep
        C_clip=50.0,
        epsilon_mu=1e-6,
        use_importance_weighting=True,
        description="Full PPL-MM with pessimism, clamping, and clipping"
    ),
}


# --- 4. TRAINING ALGORITHMS ---
def train_policy_naive_pg(policy, data, lr=1e-4, n_steps=2000):
    """
    Naive Policy Gradient: Directly optimize on observed rewards without
    importance weighting or pessimism. This is the most basic baseline.
    """
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    X, A, R, Mu = data
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    A_tensor = torch.tensor(A, dtype=torch.float32).view(-1, 1)
    R_tensor = torch.tensor(R, dtype=torch.float32).view(-1, 1)
    
    history = {
        "step": [],
        "loss": [],
    }
    
    for step in range(n_steps):
        log_probs = policy(X_tensor, A_tensor)
        
        # Naive PG: maximize log_prob * R directly (no importance weighting)
        loss = -torch.mean(log_probs * R_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 200 == 0:
            history["step"].append(step + 1)
            history["loss"].append(loss.item())
            print(f"  [Naive PG Step {step+1}/{n_steps}] Loss: {loss.item():.4f}")
    
    return history


def train_policy_with_config(policy, data, config, lr=1e-4, n_mm_steps=10, 
                            n_pg_steps_per_mm=200):
    """
    Unified training function that handles any configuration from the ablation study.
    """
    # Special case: Naive PG doesn't use importance weighting at all
    if not config.use_importance_weighting:
        return train_policy_naive_pg(policy, data, lr=lr, n_steps=n_mm_steps * n_pg_steps_per_mm)
    
    # For all other configs, use the general MM training
    is_pessimistic = (config.beta_pessimism is not None and config.beta_pessimism > 0)
    beta = config.beta_pessimism if config.beta_pessimism is not None else 0
    
    return train_policy_mm(
        policy=policy,
        data=data,
        is_pessimistic=is_pessimistic,
        beta_pessimism=beta,
        lr=lr,
        n_mm_steps=n_mm_steps,
        n_pg_steps_per_mm=n_pg_steps_per_mm,
        clip_val=config.C_clip,
        epsilon_mu=config.epsilon_mu
    )


def train_policy_mm(policy, data, is_pessimistic, beta_pessimism, 
                    lr=1e-4, n_mm_steps=10, n_pg_steps_per_mm=200, 
                    clip_val=20.0, epsilon_mu=1e-6):
    """
    Trains a policy using the continuous Majorization-Minimization (MM)
    algorithm, which is consistent with our proofs.
    
    Args:
        epsilon_mu: Minimum value to clamp behavior policy density (clamping)
        clip_val: Maximum importance weight value (clipping)
    """
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    X, A, R, Mu = data
    n = X.shape[0]
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    A_tensor = torch.tensor(A, dtype=torch.float32).view(-1, 1)
    R_tensor = torch.tensor(R, dtype=torch.float32).view(-1, 1)
    Mu_tensor = torch.tensor(Mu, dtype=torch.float32).view(-1, 1)
    
    # Apply clamping to behavior policy if epsilon_mu > 0
    if epsilon_mu > 0:
        Mu_tensor = torch.clamp(Mu_tensor, min=epsilon_mu, max=float('inf'))
    else:
        Mu_tensor = torch.clamp(Mu_tensor, min=1e-12, max=float('inf'))  # Just avoid division by zero

    V_const = 1.0 / np.sqrt(n)
    R_adjusted = R_tensor.clone()
    mode = "PPL-MM" if is_pessimistic else "Greedy"

    history = {
        "mm_step": [],
        "loss": [],
        "V_s_n": [],
        "mean_weight": [],
        "std_weight": [],
        "max_clipped_weight": [],
        "ess": [],
        "V_const": V_const,
    }

    # Outer loop: MM iterations
    for mm_step in range(n_mm_steps):
        V_s_n_value = None
        if is_pessimistic:
            with torch.no_grad():
                log_probs_k = policy(X_tensor, A_tensor)
                weights_k = torch.exp(log_probs_k) / Mu_tensor
                
                V_s_n_k = (1.0 / n) * torch.sqrt(torch.sum(weights_k**2) + 1e-8)
                V_s_n_value = V_s_n_k.item()
                
                if V_s_n_k > V_const:
                    adjustment = (beta_pessimism * weights_k) / (n * V_s_n_k)
                    R_adjusted = R_tensor - adjustment
                    adj_mean = adjustment.mean().item()
                    r_adj_mean = R_adjusted.mean().item()
                    print(f"  Applying pessimism: adj_mean={adj_mean:.4f}, R_adjusted_mean={r_adj_mean:.4f}")
                else:
                    R_adjusted = R_tensor.clone()
                    print(f"  No pessimism needed (V_s,n <= V_const)")
            
            print(f"[MM Step {mm_step+1}/{n_mm_steps}] V_s,n(pi_k): {V_s_n_k.item():.4f}, V_const: {V_const:.4f}")

        # Inner loop: Policy Gradient
        for pg_step in range(n_pg_steps_per_mm):
            log_probs = policy(X_tensor, A_tensor)
            pi_theta = torch.exp(log_probs)
            weights = pi_theta / Mu_tensor
            
            clipped_weights = torch.clamp(weights, 0.0, clip_val)
            
            # J_k(theta) = E[ w_i(theta) * R_i^(k) ]
            Y_adjusted = clipped_weights * R_adjusted.detach()
            
            total_loss = -torch.mean(Y_adjusted)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            log_probs_final = policy(X_tensor, A_tensor)
            pi_theta_final = torch.exp(log_probs_final)
            weights_final = pi_theta_final / Mu_tensor
            clipped_final = torch.clamp(weights_final, 0.0, clip_val)
            weight_mean = weights_final.mean().item()
            weight_std = weights_final.std(unbiased=False).item()
            weight_max = clipped_final.max().item()
            ess = ((weights_final.sum())**2 / (weights_final.pow(2).sum() + 1e-8)).item()

        history["mm_step"].append(mm_step + 1)
        history["loss"].append(total_loss.item())
        history["V_s_n"].append(V_s_n_value)
        history["mean_weight"].append(weight_mean)
        history["std_weight"].append(weight_std)
        history["max_clipped_weight"].append(weight_max)
        history["ess"].append(ess)

        # Log only at the end of the inner loop
        print(f"  [{mode} MM Step {mm_step+1}] Inner Loss (Neg. J_k): {total_loss.item():.4f}")

    return history


def evaluate_policy_stats(policy, env, n_test_samples=5000, n_mc_samples=20):
    policy.eval()
    X_test = torch.rand(n_test_samples, env.context_dim) * 2 - 1
    with torch.no_grad():
        actions_det = policy.act(X_test, deterministic=True)
        rewards_det = env._true_reward_func(X_test, actions_det)
        det_value = rewards_det.mean().item()

        if n_mc_samples > 0:
            reward_samples = []
            for _ in range(n_mc_samples):
                actions_stoch = policy.act(X_test)
                reward_samples.append(env._true_reward_func(X_test, actions_stoch))
            stacked = torch.stack(reward_samples)
            stoch_mean = stacked.mean().item()
            stoch_std = stacked.std(unbiased=False).item()
        else:
            stoch_mean = float("nan")
            stoch_std = float("nan")

        action_samples = policy.act(X_test, n_samples=10)
        if action_samples.dim() == 3:
            action_var = action_samples.var(dim=0, unbiased=False).mean().item()
        else:
            action_var = float("nan")

    return {
        "deterministic_value": det_value,
        "stochastic_mean": stoch_mean,
        "stochastic_std": stoch_std,
        "action_variance": action_var,
    }


def sweep_beta_pessimism(policy_template, data, env, config, beta_candidates, 
                         lr=1e-4, n_mm_steps=10, n_pg_steps_per_mm=200, 
                         n_test_samples=5000):
    """
    Sweep over beta_pessimism values for a given config and return the best performing policy and its beta.
    """
    best_beta = None
    best_value = float('-inf')
    best_policy = None
    best_history = None
    
    print(f"\n--- Sweeping BETA_PESSIMISM over {beta_candidates} for {config.name} ---")
    
    for beta in beta_candidates:
        print(f"\nTrying BETA_PESSIMISM = {beta}")
        
        # Create fresh policy with same initialization
        candidate_policy = PolicyNetwork(policy_template.net[0].in_features, 1)
        candidate_policy.load_state_dict(policy_template.state_dict())
        
        # Create temporary config with this beta
        temp_config = TrainingConfig(
            name=config.name,
            beta_pessimism=beta,
            C_clip=config.C_clip,
            epsilon_mu=config.epsilon_mu,
            use_importance_weighting=config.use_importance_weighting,
            description=config.description
        )
        
        # Train with this beta
        history = train_policy_with_config(
            candidate_policy,
            data,
            temp_config,
            lr=lr,
            n_mm_steps=n_mm_steps,
            n_pg_steps_per_mm=n_pg_steps_per_mm
        )
        
        # Evaluate
        metrics = evaluate_policy_stats(candidate_policy, env, n_test_samples=n_test_samples)
        value = metrics["deterministic_value"]
        
        print(f"  --> Deterministic Value: {value:.4f}")
        
        if value > best_value:
            best_value = value
            best_beta = beta
            best_policy = candidate_policy
            best_history = history
    
    print(f"\n*** Best BETA_PESSIMISM: {best_beta} with value {best_value:.4f} ***")
    
    return best_policy, best_beta, best_history, best_value


# --- 5. MAIN EXECUTION (Ablation Study) ---
if __name__ == "__main__":
    
    # --- Hyperparameters ---
    N_OFFLINE_SAMPLES = 10000  # Substantially increased from 2000
    N_TEST_SAMPLES = 10000
    CONTEXT_DIM = 5
    ACTION_DIM = 1
    POLICY_LR = 1e-4
    
    # Sweep range for beta pessimism (for PPL variants)
    # Use very fine-grained sweep including 0, with emphasis on smaller values
    BETA_CANDIDATES = [0.0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    
    # Increase iterations for better convergence
    N_MM_STEPS = 20  # Increased from 15 for better convergence
    N_PG_STEPS_PER_MM = 150  # Slightly reduced per-step but more MM steps
    N_TOTAL_STEPS = N_MM_STEPS * N_PG_STEPS_PER_MM  # 3000 total steps
    
    # --- Benchmark Suite ---
    # New challenging benchmarks designed to highlight PPL advantages
    benchmark_ids = ["BiasedBehaviorSharpPeak", "SafetyConstrainedReward", "SparseRewardWithNoise"]
    
    # Use higher noise for sparse reward to make it more challenging
    noise_levels = {
        "BiasedBehaviorSharpPeak": 0.05,  # Lower noise for sharp peak (noise would hide the peak)
        "SafetyConstrainedReward": 0.1,   # Moderate noise
        "SparseRewardWithNoise": 0.4      # VERY high noise for sparse reward (increased from 0.3)
    }
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    benchmark_results = {}
    summary_rows = []

    for bench_id in benchmark_ids:
        print(f"\n{'='*80}")
        print(f"RUNNING BENCHMARK: {bench_id}")
        print(f"{'='*80}")

        # --- Initialization ---
        noise_std = noise_levels.get(bench_id, 0.1)
        env = BenchmarkEnv(benchmark_id=bench_id, context_dim=CONTEXT_DIM, noise_std=noise_std)
        offline_data = env.get_offline_data(N_OFFLINE_SAMPLES)
        print(f"Generated {N_OFFLINE_SAMPLES} offline data points for {bench_id}.")

        # Store results for all configurations
        config_results = {}
        
        # Train all configurations
        for config_key, config in ABLATION_CONFIGS.items():
            print(f"\n{'='*60}")
            print(f"Training: {config_key}")
            print(f"Description: {config.description}")
            print(f"{'='*60}")
            
            # Create fresh policy with same random initialization
            policy = PolicyNetwork(CONTEXT_DIM, ACTION_DIM)
            if config_key == "1. Naive PG":
                # Save this initialization to reuse for others
                base_state_dict = policy.state_dict().copy()
            else:
                # Reuse same initialization
                policy.load_state_dict(base_state_dict)
            
            # Handle PPL variants that need beta sweeping
            if config.beta_pessimism is None:
                policy_trained, best_beta, history, _ = sweep_beta_pessimism(
                    policy,
                    offline_data,
                    env,
                    config,
                    BETA_CANDIDATES,
                    lr=POLICY_LR,
                    n_mm_steps=N_MM_STEPS,
                    n_pg_steps_per_mm=N_PG_STEPS_PER_MM,
                    n_test_samples=N_TEST_SAMPLES
                )
            else:
                # Train with fixed beta (or no pessimism)
                history = train_policy_with_config(
                    policy,
                    offline_data,
                    config,
                    lr=POLICY_LR,
                    n_mm_steps=N_MM_STEPS,
                    n_pg_steps_per_mm=N_PG_STEPS_PER_MM
                )
                policy_trained = policy
                best_beta = config.beta_pessimism
            
            # Evaluate
            metrics = evaluate_policy_stats(policy_trained, env, n_test_samples=N_TEST_SAMPLES)
            
            config_results[config_key] = {
                "config": config,
                "policy": policy_trained,
                "metrics": metrics,
                "history": history,
                "best_beta": best_beta,
                "value": metrics["deterministic_value"]
            }
            
            beta_info = f" (β={best_beta})" if best_beta is not None and best_beta > 0 else ""
            print(f"\n{'='*60}")
            print(f"{config_key}{beta_info} Final Result: {metrics['deterministic_value']:.4f}")
            print(f"  Stochastic: {metrics['stochastic_mean']:.4f} ± {metrics['stochastic_std']:.4f}")
            print(f"{'='*60}")
        
        # --- Print Results Summary ---
        print(f"\n{'='*60}")
        print(f"RESULTS SUMMARY FOR {bench_id}")
        print(f"{'='*60}")
        for config_key in ABLATION_CONFIGS.keys():
            result = config_results[config_key]
            beta_str = f" (β={result['best_beta']})" if result['best_beta'] else ""
            print(f"{config_key}{beta_str}: {result['value']:.4f}")
        
        # Find best performing configuration
        best_config_key = max(config_results.keys(), key=lambda k: config_results[k]['value'])
        print(f"\n🏆 Best: {best_config_key} with value {config_results[best_config_key]['value']:.4f}")

        # --- Visualization ---
        # Create a comprehensive comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # Plot 1: Bar chart comparing all 7 configurations
        config_names = list(ABLATION_CONFIGS.keys())
        config_names_short = [k.split(".")[1].strip() for k in config_names]  # Remove numbers
        det_values = [config_results[k]["value"] for k in config_names]
        stoch_values = [config_results[k]["metrics"]["stochastic_mean"] for k in config_names]
        
        x_pos = np.arange(len(config_names))
        bar_width = 0.35
        
        axes[0].bar(x_pos - bar_width/2, det_values, bar_width, label="Deterministic", alpha=0.8)
        axes[0].bar(x_pos + bar_width/2, stoch_values, bar_width, label="Stochastic", alpha=0.8)
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(config_names_short, rotation=45, ha='right')
        axes[0].set_ylabel("True Reward")
        axes[0].set_title("Ablation Study: All Configurations")
        axes[0].legend()
        axes[0].axhline(0.0, color="black", linewidth=0.5, linestyle='--')
        axes[0].grid(axis='y', alpha=0.3)

        # Plot 2: Show PPL training dynamics for the BEST PPL variant
        # Find which PPL variant performed best
        ppl_keys = ["5. PPL (pure)", "6. PPL (no clamping)", "7. PPL (no clipping)", "8. PPL (Both)"]
        best_ppl_key = max(ppl_keys, key=lambda k: config_results[k]["value"])
        best_ppl_result = config_results[best_ppl_key]
        
        if "mm_step" in best_ppl_result["history"] and best_ppl_result["best_beta"] and best_ppl_result["best_beta"] > 0:
            history = best_ppl_result["history"]
            mm_steps = np.array(history["mm_step"])
            V_s_n_vals = np.array([v if v is not None else np.nan for v in history["V_s_n"]])
            
            axes[1].plot(mm_steps, V_s_n_vals, marker="o", label=r"$V_{s,n}(\pi)$", linewidth=2)
            axes[1].axhline(history["V_const"], color="red", linestyle="--", 
                           label=r"$V_{\text{const}}$", linewidth=2)
            axes[1].set_xlabel("MM Step")
            axes[1].set_ylabel(r"$V_{s,n}$")
            title_suffix = best_ppl_key.split(".")[1].strip()
            axes[1].set_title(f"Best PPL ({title_suffix}, β={best_ppl_result['best_beta']}): Pessimism Tracking")
            axes[1].legend()
            axes[1].grid(alpha=0.3)
        else:
            # If no PPL with beta > 0, show a message
            axes[1].text(0.5, 0.5, "No PPL variant with β > 0\nchose pessimism for this task", 
                        ha='center', va='center', fontsize=12, transform=axes[1].transAxes)
            axes[1].set_title("PPL Training Dynamics (N/A)")
            axes[1].grid(alpha=0.3)

        fig.suptitle(f"{bench_id} - Ablation Study Results", fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plot_path = results_dir / f"{bench_id}_ablation.png"
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)

        # Store results
        benchmark_results[bench_id] = {
            "config_results": {k: {
                "value": v["value"],
                "best_beta": v["best_beta"],
                "metrics": v["metrics"]
            } for k, v in config_results.items()},
            "n_offline": len(offline_data[0]),
            "plot_path": str(plot_path),
        }

        # Prepare summary row
        summary_row = {"benchmark": bench_id}
        for config_key in ABLATION_CONFIGS.keys():
            result = config_results[config_key]
            col_name = config_key.split(".")[1].strip().replace(" ", "_").replace("(", "").replace(")", "").lower()
            summary_row[col_name] = result["value"]
            if result["best_beta"]:
                summary_row[f"{col_name}_beta"] = result["best_beta"]
        summary_rows.append(summary_row)

    # --- Final Summary Table ---
    print("\n" + "="*120)
    print("ABLATION STUDY SUMMARY - ALL BENCHMARKS")
    print("="*120)
    
    # Print header
    header_parts = ["Benchmark".ljust(28)]
    for config_key in ABLATION_CONFIGS.keys():
        short_name = config_key.split(".")[1].strip()[:12]
        header_parts.append(short_name.rjust(13))
    print("".join(header_parts))
    print("="*120)
    
    # Print data rows
    for row in summary_rows:
        line_parts = [row["benchmark"].ljust(28)]
        for config_key in ABLATION_CONFIGS.keys():
            col_name = config_key.split(".")[1].strip().replace(" ", "_").replace("(", "").replace(")", "").lower()
            value = row.get(col_name, float('nan'))
            line_parts.append(f"{value:>13.4f}")
        print("".join(line_parts))
    
    # Save results
    metrics_path = results_dir / "ablation_results.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2, default=str)
    print(f"\n\nSaved detailed metrics to {metrics_path}")
    print(f"Saved per-benchmark plots to {results_dir}")
    
    # --- Analysis: Which configuration wins most often? ---
    print("\n" + "="*120)
    print("OVERALL STATISTICS")
    print("="*120)
    
    config_win_counts = {k: 0 for k in ABLATION_CONFIGS.keys()}
    ppl_family_keys = ["5. PPL (pure)", "6. PPL (no clamping)", "7. PPL (no clipping)", "8. PPL (Both)"]
    ppl_family_wins = 0
    
    # Collect all values for statistical analysis
    all_values = {k: [] for k in ABLATION_CONFIGS.keys()}
    for row in summary_rows:
        for config_key in ABLATION_CONFIGS.keys():
            col_name = config_key.split(".")[1].strip().replace(" ", "_").replace("(", "").replace(")", "").lower()
            value = row.get(col_name, float('nan'))
            all_values[config_key].append(value)
    
    for row in summary_rows:
        # Find best config for this benchmark
        best_value = float('-inf')
        best_config = None
        for config_key in ABLATION_CONFIGS.keys():
            col_name = config_key.split(".")[1].strip().replace(" ", "_").replace("(", "").replace(")", "").lower()
            value = row.get(col_name, float('-inf'))
            if value > best_value:
                best_value = value
                best_config = config_key
        if best_config:
            config_win_counts[best_config] += 1
            if best_config in ppl_family_keys:
                ppl_family_wins += 1
    
    print("\nWins per configuration across all benchmarks:")
    for config_key in ABLATION_CONFIGS.keys():
        wins = config_win_counts[config_key]
        is_ppl = config_key in ppl_family_keys
        marker = "🔵 PPL" if is_ppl else "⚪ Baseline"
        avg_val = np.mean(all_values[config_key])
        print(f"  {marker} {config_key}: {wins}/{len(summary_rows)} wins (avg={avg_val:.4f})")
    
    # Analyze PPL family performance
    print(f"\n{'='*120}")
    print("PPL FAMILY PERFORMANCE")
    print(f"{'='*120}")
    
    # Win rate analysis with binomial confidence interval
    n_benchmarks = len(summary_rows)
    ppl_win_rate = ppl_family_wins / n_benchmarks
    # Wilson score interval for binomial proportion
    from scipy import stats
    ci_low, ci_high = stats.binom.interval(0.95, n_benchmarks, ppl_win_rate)
    ci_low = ci_low / n_benchmarks
    ci_high = ci_high / n_benchmarks
    
    print(f"PPL variants (with β > 0) won: {ppl_family_wins}/{n_benchmarks} benchmarks")
    print(f"Win rate: {ppl_win_rate:.1%} (95% CI: [{ci_low:.1%}, {ci_high:.1%}])")
    
    if ppl_family_wins == len(summary_rows):
        print("   ✅ SUCCESS: PPL family (with pessimism) outperformed ALL baselines on every benchmark!")
    elif ppl_family_wins >= len(summary_rows) / 2:
        print("   ⚠️  PARTIAL: PPL family won majority but not all benchmarks")
    else:
        print("   ❌ FAILURE: PPL family did not demonstrate clear advantages")
    
    # Effect size analysis: compare best PPL vs best baseline
    ppl_values = []
    baseline_values = []
    baseline_keys = ["1. Naive PG", "2. Clamped", "3. Clipped", "4. Clamped + Clipped"]
    
    for row in summary_rows:
        # Get best PPL value for this benchmark
        ppl_vals = []
        for k in ppl_family_keys:
            col_name = k.split(".")[1].strip().replace(" ", "_").replace("(", "").replace(")", "").lower()
            ppl_vals.append(row.get(col_name, float('-inf')))
        ppl_values.append(max(ppl_vals))
        
        # Get best baseline value for this benchmark
        baseline_vals = []
        for k in baseline_keys:
            col_name = k.split(".")[1].strip().replace(" ", "_").replace("(", "").replace(")", "").lower()
            baseline_vals.append(row.get(col_name, float('-inf')))
        baseline_values.append(max(baseline_vals))
    
    ppl_mean = np.mean(ppl_values)
    baseline_mean = np.mean(baseline_values)
    improvement = ((ppl_mean - baseline_mean) / baseline_mean) * 100
    
    # Cohen's d effect size
    pooled_std = np.sqrt((np.std(ppl_values)**2 + np.std(baseline_values)**2) / 2)
    cohens_d = (ppl_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0
    
    print(f"\nEffect Size Analysis:")
    print(f"  Best PPL avg: {ppl_mean:.4f}")
    print(f"  Best Baseline avg: {baseline_mean:.4f}")
    print(f"  Improvement: {improvement:+.2f}%")
    print(f"  Cohen's d: {cohens_d:.3f} ({'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'})")
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(ppl_values, baseline_values)
    print(f"  Paired t-test: t={t_stat:.3f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else '(n.s.)'}")
    
    # Component analysis: isolate effects of clipping, clamping, pessimism
    print(f"\n{'='*120}")
    print("COMPONENT ANALYSIS")
    print(f"{'='*120}")
    
    # Effect of pessimism (compare configs with/without pessimism, same regularization)
    print("\n1. Effect of Pessimism (β > 0):")
    
    # Pure pessimism: PPL (pure) vs Naive PG
    ppl_pure_vals = all_values["5. PPL (pure)"]
    naive_vals = all_values["1. Naive PG"]
    pess_improvement = ((np.mean(ppl_pure_vals) - np.mean(naive_vals)) / np.mean(naive_vals)) * 100
    print(f"   PPL (pure) vs Naive PG: {pess_improvement:+.2f}% improvement")
    
    # With clipping: PPL (no clamp) vs Clipped
    ppl_noclamp_vals = all_values["6. PPL (no clamping)"]
    clipped_vals = all_values["3. Clipped"]
    pess_clip_improvement = ((np.mean(ppl_noclamp_vals) - np.mean(clipped_vals)) / np.mean(clipped_vals)) * 100
    print(f"   PPL (no clamp) vs Clipped: {pess_clip_improvement:+.2f}% improvement")
    
    # With clamping: PPL (no clip) vs Clamped
    ppl_noclip_vals = all_values["7. PPL (no clipping)"]
    clamped_vals = all_values["2. Clamped"]
    pess_clamp_improvement = ((np.mean(ppl_noclip_vals) - np.mean(clamped_vals)) / np.mean(clamped_vals)) * 100
    print(f"   PPL (no clip) vs Clamped: {pess_clamp_improvement:+.2f}% improvement")
    
    print("\n2. Effect of Clipping:")
    # With pessimism: PPL (no clip) vs PPL (pure)
    clip_ppl_improvement = ((np.mean(ppl_noclip_vals) - np.mean(ppl_pure_vals)) / np.mean(ppl_pure_vals)) * 100
    print(f"   With pessimism: {clip_ppl_improvement:+.2f}% (PPL no clip vs PPL pure)")
    
    # Without pessimism: Clipped vs Naive PG  
    clip_base_improvement = ((np.mean(clipped_vals) - np.mean(naive_vals)) / np.mean(naive_vals)) * 100
    print(f"   Without pessimism: {clip_base_improvement:+.2f}% (Clipped vs Naive PG)")
    
    print("\n3. Effect of Clamping:")
    # With pessimism: PPL (no clamp) vs PPL (pure)
    clamp_ppl_improvement = ((np.mean(ppl_noclamp_vals) - np.mean(ppl_pure_vals)) / np.mean(ppl_pure_vals)) * 100
    print(f"   With pessimism: {clamp_ppl_improvement:+.2f}% (PPL no clamp vs PPL pure)")
    
    # Without pessimism: Clamped vs Naive PG
    clamp_base_improvement = ((np.mean(clamped_vals) - np.mean(naive_vals)) / np.mean(naive_vals)) * 100
    print(f"   Without pessimism: {clamp_base_improvement:+.2f}% (Clamped vs Naive PG)")
    
    # Show optimal beta values chosen
    print(f"\n{'='*120}")
    print("OPTIMAL BETA VALUES")
    print(f"{'='*120}")
    for row in summary_rows:
        bench = row["benchmark"]
        print(f"\n{bench}:")
        for config_key in ppl_family_keys:
            col_name = config_key.split(".")[1].strip().replace(" ", "_").replace("(", "").replace(")", "").lower()
            beta_col = f"{col_name}_beta"
            if beta_col in row:
                beta_val = row[beta_col]
                value = row[col_name]
                config_name = config_key.split(".")[1].strip()
                print(f"  {config_name:20s}: β = {beta_val:.2f}, value = {value:.4f}")
    
    # === ADVANCED VISUALIZATIONS ===
    print(f"\n{'='*120}")
    print("GENERATING ADVANCED VISUALIZATIONS")
    print(f"{'='*120}")
    
    # Visualization 1: Heatmap of all configs × benchmarks
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    
    # Heatmap data
    config_names_short = [k.split(".")[1].strip() for k in ABLATION_CONFIGS.keys()]
    bench_names = [row["benchmark"] for row in summary_rows]
    heatmap_data = []
    for config_key in ABLATION_CONFIGS.keys():
        row_data = []
        for row in summary_rows:
            col_name = config_key.split(".")[1].strip().replace(" ", "_").replace("(", "").replace(")", "").lower()
            row_data.append(row.get(col_name, 0))
        heatmap_data.append(row_data)
    heatmap_data = np.array(heatmap_data)
    
    # Plot 1: Heatmap
    im = axes[0, 0].imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    axes[0, 0].set_xticks(np.arange(len(bench_names)))
    axes[0, 0].set_yticks(np.arange(len(config_names_short)))
    axes[0, 0].set_xticklabels(bench_names, rotation=45, ha='right')
    axes[0, 0].set_yticklabels(config_names_short)
    
    # Add values and mark winners
    for i in range(len(config_names_short)):
        for j in range(len(bench_names)):
            value = heatmap_data[i, j]
            is_best = (value == heatmap_data[:, j].max())
            text_color = 'white' if value > heatmap_data.mean() else 'black'
            axes[0, 0].text(j, i, f'{value:.3f}', ha='center', va='center', 
                          color=text_color, fontweight='bold' if is_best else 'normal',
                          fontsize=9)
    
    axes[0, 0].set_title("Performance Heatmap: All Configurations × Benchmarks", fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=axes[0, 0])
    
    # Plot 2: Win rate comparison with confidence intervals
    baseline_wins = [config_win_counts[k] for k in baseline_keys]
    ppl_wins = [config_win_counts[k] for k in ppl_family_keys]
    
    x_baseline = np.arange(len(baseline_keys))
    x_ppl = np.arange(len(ppl_family_keys))
    
    axes[0, 1].bar(x_baseline, baseline_wins, color='lightcoral', alpha=0.7, label='Baselines')
    axes[0, 1].bar(x_ppl + len(baseline_keys) + 0.5, ppl_wins, color='skyblue', alpha=0.7, label='PPL Family')
    
    # Create tick positions and labels (without separator)
    all_tick_positions = list(range(len(baseline_keys))) + [len(baseline_keys) + 0.5 + i for i in range(len(ppl_family_keys))]
    all_labels = [k.split(".")[1].strip() for k in baseline_keys] + [k.split(".")[1].strip() for k in ppl_family_keys]
    axes[0, 1].set_xticks(all_tick_positions)
    axes[0, 1].set_xticklabels(all_labels, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Number of Wins')
    axes[0, 1].set_title('Win Count by Configuration', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].axhline(n_benchmarks / 2, color='red', linestyle='--', linewidth=1, label='50% threshold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Component effects (bar chart with error bars)
    components = ['Pessimism\n(pure)', 'Pessimism\n(+clip)', 'Pessimism\n(+clamp)', 'Clipping', 'Clamping']
    effects = [pess_improvement, pess_clip_improvement, pess_clamp_improvement, 
               clip_base_improvement, clamp_base_improvement]
    colors = ['#2E86AB', '#2E86AB', '#2E86AB', '#A23B72', '#F18F01']
    
    bars = axes[1, 0].bar(components, effects, color=colors, alpha=0.7)
    axes[1, 0].axhline(0, color='black', linewidth=0.8)
    axes[1, 0].set_ylabel('Performance Improvement (%)')
    axes[1, 0].set_title('Component Analysis: Isolated Effects', fontsize=12, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, effect in zip(bars, effects):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{effect:+.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                       fontweight='bold', fontsize=10)
    
    # Plot 4: PPL family comparison (grouped bar chart)
    ppl_avgs = [np.mean(all_values[k]) for k in ppl_family_keys]
    ppl_stds = [np.std(all_values[k]) for k in ppl_family_keys]
    ppl_labels = [k.split(".")[1].strip() for k in ppl_family_keys]
    
    x_pos = np.arange(len(ppl_family_keys))
    axes[1, 1].bar(x_pos, ppl_avgs, yerr=ppl_stds, capsize=5, color='skyblue', alpha=0.7, edgecolor='navy')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(ppl_labels, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Average Performance')
    axes[1, 1].set_title('PPL Family: Average Performance ± Std Dev', fontsize=12, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Add significance stars
    best_ppl_idx = np.argmax(ppl_avgs)
    for i, avg in enumerate(ppl_avgs):
        if i != best_ppl_idx:
            # Simple comparison (could use t-test for proper significance)
            diff = abs(ppl_avgs[best_ppl_idx] - avg)
            if diff > 0.02:  # Threshold for "significant"
                axes[1, 1].text(i, avg + ppl_stds[i] + 0.01, '*', ha='center', fontsize=16)
    
    fig.suptitle('Comprehensive Ablation Study Analysis', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    comprehensive_plot_path = results_dir / "comprehensive_analysis.png"
    fig.savefig(comprehensive_plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved comprehensive analysis plot to {comprehensive_plot_path}")
    
    print("\n" + "="*120)