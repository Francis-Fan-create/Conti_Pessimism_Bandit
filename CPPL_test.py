import json
from pathlib import Path

import matplotlib.pyplot as plt
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
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    benchmark_results = {}
    summary_rows = []

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
        greedy_history = train_policy_mm(
            greedy_policy, 
            offline_data, 
            is_pessimistic=False,
            beta_pessimism=0,
            lr=POLICY_LR,
            n_mm_steps=1, 
            n_pg_steps_per_mm=N_PG_STEPS_GREEDY
        )
        
        print("\n--- Training Pessimistic Policy (PPL-MM) ---")
        ppl_history = train_policy_mm(
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
        greedy_metrics = evaluate_policy_stats(greedy_policy, env, n_test_samples=N_TEST_SAMPLES)
        ppl_metrics = evaluate_policy_stats(ppl_policy, env, n_test_samples=N_TEST_SAMPLES)
        val_greedy = greedy_metrics["deterministic_value"]
        val_ppl = ppl_metrics["deterministic_value"]
        
        print("\n--- Results ---")
        print(f"True Value of Greedy Policy: {val_greedy:.4f}")
        print(f"True Value of PPL-MM Policy: {val_ppl:.4f}")
        print(f"Stochastic Value (mean ± std) Greedy: {greedy_metrics['stochastic_mean']:.4f} ± {greedy_metrics['stochastic_std']:.4f}")
        print(f"Stochastic Value (mean ± std) PPL-MM: {ppl_metrics['stochastic_mean']:.4f} ± {ppl_metrics['stochastic_std']:.4f}")
        print(f"Action variance Greedy: {greedy_metrics['action_variance']:.4e}")
        print(f"Action variance PPL-MM: {ppl_metrics['action_variance']:.4e}")

        if val_ppl > val_greedy:
            print("\nPPL-MM successfully found a better policy.")
        else:
            print("\nPPL-MM did not outperform the greedy policy in this run.")

        # --- Visualization ---
        mm_steps = np.array(ppl_history["mm_step"])
        V_s_n_vals = np.array([v if v is not None else np.nan for v in ppl_history["V_s_n"]])
        ess_vals = np.array(ppl_history["ess"])
        std_vals = np.array(ppl_history["std_weight"])
        ess_ratio = ess_vals / len(offline_data[0])

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        x_pos = np.arange(2)
        bar_width = 0.35
        det_values = [greedy_metrics["deterministic_value"], ppl_metrics["deterministic_value"]]
        stoch_values = [greedy_metrics["stochastic_mean"], ppl_metrics["stochastic_mean"]]
        axes[0].bar(x_pos - bar_width/2, det_values, bar_width, label="Deterministic")
        axes[0].bar(x_pos + bar_width/2, stoch_values, bar_width, label="Stochastic")
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(["Greedy", "PPL-MM"])
        axes[0].set_ylabel("True Reward")
        axes[0].set_title("Policy Value Comparison")
        axes[0].legend()
        axes[0].axhline(0.0, color="black", linewidth=0.5)

        axes[1].plot(mm_steps, V_s_n_vals, marker="o", label=r"$V_{s,n}(\pi)$")
        axes[1].axhline(ppl_history["V_const"], color="red", linestyle="--", label=r"$V_{\text{const}}$")
        axes[1].set_xlabel("MM Step")
        axes[1].set_ylabel(r"$V_{s,n}$")
        axes[1].set_title("Pessimism Constraint Tracking")
        axes[1].legend()

        ax3 = axes[2]
        ax3.plot(mm_steps, ess_ratio, marker="o", color="tab:blue", label="ESS / n")
        ax3.set_xlabel("MM Step")
        ax3.set_ylabel("ESS / n", color="tab:blue")
        ax3.tick_params(axis="y", labelcolor="tab:blue")
        ax3.set_title("Importance Weight Diagnostics")
        ax3b = ax3.twinx()
        ax3b.plot(mm_steps, std_vals, marker="s", color="tab:orange", label="Weight Std")
        ax3b.set_ylabel("Weight Std", color="tab:orange")
        ax3b.tick_params(axis="y", labelcolor="tab:orange")
        lines_ax3, labels_ax3 = ax3.get_legend_handles_labels()
        lines_ax3b, labels_ax3b = ax3b.get_legend_handles_labels()
        ax3.legend(lines_ax3 + lines_ax3b, labels_ax3 + labels_ax3b, loc="upper right")

        fig.suptitle(f"{bench_id} Metrics", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        plot_path = results_dir / f"{bench_id}_metrics.png"
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)

        benchmark_results[bench_id] = {
            "greedy_metrics": greedy_metrics,
            "ppl_metrics": ppl_metrics,
            "greedy_history": greedy_history,
            "ppl_history": ppl_history,
            "n_offline": len(offline_data[0]),
            "plot_path": str(plot_path),
        }

        summary_rows.append({
            "benchmark": bench_id,
            "greedy_det": val_greedy,
            "ppl_det": val_ppl,
            "delta_det": val_ppl - val_greedy,
            "greedy_stoch": greedy_metrics["stochastic_mean"],
            "ppl_stoch": ppl_metrics["stochastic_mean"],
            "ppl_stoch_std": ppl_metrics["stochastic_std"],
            "ppl_action_var": ppl_metrics["action_variance"],
            "min_ess_ratio": float(ess_ratio.min()),
            "max_weight_std": float(std_vals.max()),
        })

    print("\n================ SUMMARY ===============")
    header = f"{'Benchmark':<28}{'Greedy':>12}{'PPL-MM':>12}{'Gain':>12}{'ESS_min':>12}{'Wstd_max':>12}"
    print(header)
    for row in summary_rows:
        print(
            f"{row['benchmark']:<28}"
            f"{row['greedy_det']:>12.4f}"
            f"{row['ppl_det']:>12.4f}"
            f"{row['delta_det']:>12.4f}"
            f"{row['min_ess_ratio']:>12.3f}"
            f"{row['max_weight_std']:>12.3f}"
        )

    metrics_path = results_dir / "benchmark_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\nSaved detailed metrics to {metrics_path}")
    print(f"Saved per-benchmark plots to {results_dir}")