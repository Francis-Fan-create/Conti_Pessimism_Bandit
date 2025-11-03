import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import numpy as np

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
        
        if benchmark_id == "TrigWithBetaHole":
            self._true_reward_func = self._true_reward_trig
            self._get_behavior_data = self._get_behavior_beta_hole
        elif benchmark_id == "SimplePeakWithMoGHole":
            self._true_reward_func = self._true_reward_simple_peak
            self._get_behavior_data = self._get_behavior_mog_hole
        elif benchmark_id == "SimpleRewardWithMovingMean":
            self._true_reward_func = self._true_reward_simple_linear
            self._get_behavior_data = self._get_behavior_moving_mean
        else:
            raise ValueError(f"Unknown benchmark_id: {benchmark_id}")

    # --- Reward Functions ---
    def _true_reward_trig(self, x, a):
        # Benchmark 1: Complex Reward
        term1 = x[:, 0].unsqueeze(1) * a
        term2 = x[:, 1].unsqueeze(1) * torch.cos(4 * np.pi * a)
        term3 = x[:, 2].unsqueeze(1) * torch.sin(4 * np.pi * a)
        return term1 + term2 + term3

    def _true_reward_simple_peak(self, x, a):
        # Benchmark 2: Simple Reward, optimal action a*(x) = x1
        a_star = x[:, 0].unsqueeze(1)
        return -(a - a_star)**2 + 1.0 # Max value is 1.0

    def _true_reward_simple_linear(self, x, a):
        # Benchmark 3: Simple Reward, optimal action a*(x) = 1
        return a

    # --- Behavior Policy Data Generation (Sample + Density) ---
    def _get_behavior_beta_hole(self, X):
        # Behavior for Benchmark 1: Beta(0.5, 0.5) "U-shape"
        n_samples = X.shape[0]
        behavior_dist = dist.Beta(torch.tensor(0.5), torch.tensor(0.5))
        A_raw = behavior_dist.sample((n_samples,))
        A = 2.0 * A_raw - 1.0 # Scale to [-1, 1]
        A = A.unsqueeze(1)
        
        log_prob_raw = behavior_dist.log_prob(A_raw)
        log_prob_mu = log_prob_raw - np.log(2.0) # Correct for scaling
        mu_prob = torch.exp(log_prob_mu).unsqueeze(1)
        return A, mu_prob

    def _get_behavior_mog_hole(self, X):
        # Behavior for Benchmark 2: Mixture of Gaussians (hole at a=0)
        n_samples = X.shape[0]
        std = torch.full((n_samples,), 0.1)
        idx = torch.randint(0, 2, (n_samples,))
        mean = torch.where(idx == 0, torch.full((n_samples,), -0.5), torch.full((n_samples,), 0.5)).float()

        A = sample_truncated_normal(mean, std)
        log_pdf1 = truncated_normal_log_pdf(A, torch.full_like(A, -0.5), std)
        log_pdf2 = truncated_normal_log_pdf(A, torch.full_like(A, 0.5), std)
        mu_prob = 0.5 * torch.exp(log_pdf1) + 0.5 * torch.exp(log_pdf2)
        return A.unsqueeze(1), mu_prob.unsqueeze(1)

    def _get_behavior_moving_mean(self, X):
        # Behavior for Benchmark 3: Mean moves with context x1
        n_samples = X.shape[0]
        std = torch.full((n_samples,), 0.1)
        mean = X[:, 0] # Mean = context x1

        A = sample_truncated_normal(mean, std)
        log_pdf = truncated_normal_log_pdf(A, mean, std)
        mu_prob = torch.exp(log_pdf)
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
        
        # Normalize R to [0, 1] for PPL-MM stability (as in paper)
        R = (R - R.min()) / (R.max() - R.min()) 
        
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

# --- 3. TRAINING ALGORITHM (PPL-MM) (No changes) ---
def train_policy_mm(policy, data, is_pessimistic, beta_pessimism, 
                    lr=1e-4, n_mm_steps=10, n_pg_steps_per_mm=200, clip_val=20.0):
    """
    Trains a policy using the continuous Majorization-Minimization (MM)
    algorithm, which is consistent with our proofs.
    """
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    X, A, R, Mu = data
    n = X.shape[0]
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    A_tensor = torch.tensor(A, dtype=torch.float32).view(-1, 1)
    R_tensor = torch.tensor(R, dtype=torch.float32).view(-1, 1)
    Mu_tensor = torch.tensor(Mu, dtype=torch.float32).view(-1, 1)
    Mu_tensor = torch.clamp(Mu_tensor, 1e-6, float('inf')) # Avoid division by zero

    V_const = 1.0 / np.sqrt(n)
    R_adjusted = R_tensor.clone()
    mode = "PPL-MM" if is_pessimistic else "Greedy"

    # Outer loop: MM iterations
    for mm_step in range(n_mm_steps):
        if is_pessimistic:
            with torch.no_grad():
                log_probs_k = policy(X_tensor, A_tensor)
                weights_k = torch.exp(log_probs_k) / Mu_tensor
                
                V_s_n_k = (1.0 / n) * torch.sqrt(torch.sum(weights_k**2) + 1e-8)
                
                if V_s_n_k > V_const:
                    adjustment = (beta_pessimism * weights_k) / (n * V_s_n_k)
                    R_adjusted = R_tensor - adjustment
                else:
                    R_adjusted = R_tensor.clone()
            
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
        
        # Log only at the end of the inner loop
        print(f"  [{mode} MM Step {mm_step+1}] Inner Loss (Neg. J_k): {total_loss.item():.4f}")

# --- 4. MAIN EXECUTION (Revised) ---
if __name__ == "__main__":
    
    # --- Hyperparameters ---
    N_OFFLINE_SAMPLES = 2000
    N_TEST_SAMPLES = 10000
    CONTEXT_DIM = 5
    ACTION_DIM = 1
    POLICY_LR = 1e-4
    
    BETA_PESSIMISM = 0.5 
    N_MM_STEPS = 10         
    N_PG_STEPS_PER_MM = 200 
    N_PG_STEPS_GREEDY = N_MM_STEPS * N_PG_STEPS_PER_MM # Match total compute
    WEIGHT_CLIP_VALUE = 20.0
    
    # --- Benchmark Suite ---
    benchmark_ids = ["TrigWithBetaHole", "SimplePeakWithMoGHole", "SimpleRewardWithMovingMean"]
    
    for bench_id in benchmark_ids:
        print(f"\n{'='*60}")
        print(f"RUNNING BENCHMARK: {bench_id}")
        print(f"{'='*60}")

        # --- Initialization ---
        env = BenchmarkEnv(benchmark_id=bench_id, context_dim=CONTEXT_DIM)
        offline_data = env.get_offline_data(N_OFFLINE_SAMPLES)
        print(f"Generated {N_OFFLINE_SAMPLES} offline data points for {bench_id}.")

        greedy_policy = PolicyNetwork(CONTEXT_DIM, ACTION_DIM)
        ppl_policy = PolicyNetwork(CONTEXT_DIM, ACTION_DIM)
        ppl_policy.load_state_dict(greedy_policy.state_dict())

        # --- Training ---
        print("\n--- Training Greedy Policy (Baseline) ---")
        train_policy_mm(
            greedy_policy, 
            offline_data, 
            is_pessimistic=False,
            beta_pessimism=0,
            lr=POLICY_LR,
            n_mm_steps=1, 
            n_pg_steps_per_mm=N_PG_STEPS_GREEDY
        )
        
        print("\n--- Training Pessimistic Policy (PPL-MM) ---")
        train_policy_mm(
            ppl_policy, 
            offline_data, 
            is_pessimistic=True,
            beta_pessimism=BETA_PESSIMISM,
            lr=POLICY_LR,
            n_mm_steps=N_MM_STEPS,
            n_pg_steps_per_mm=N_PG_STEPS_PER_MM
        )

        # --- Evaluation ---
        print(f"\n--- Evaluating Policies for {bench_id} ---")
        val_greedy = env.evaluate_policy(greedy_policy, n_test_samples=N_TEST_SAMPLES)
        val_ppl = env.evaluate_policy(ppl_policy, n_test_samples=N_TEST_SAMPLES)
        
        print("\n--- Results ---")
        print(f"True Value of Greedy Policy: {val_greedy:.4f}")
        print(f"True Value of PPL-MM Policy: {val_ppl:.4f}")

        if val_ppl > val_greedy:
            print("\nPPL-MM successfully found a better policy.")
        else:
            print("\nPPL-MM did not outperform the greedy policy in this run.")