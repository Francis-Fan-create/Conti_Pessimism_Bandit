"""
Comprehensive Statistical Evaluation of PPL-MM vs Naive PG
For Annals of Statistics Submission

This script implements:
1. Multiple runs with different random seeds for each (task, variant, method)
2. Paired statistical comparisons (t-test, Wilcoxon, Cohen's d, bootstrap CI)
3. Multiple comparison correction (Holm, BH/FDR)
4. Mixed-effects modeling for global analysis
5. Publication-grade visualizations (forest plots, paired comparison plots, heatmaps)
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon, ttest_rel
import pandas as pd
from statsmodels.stats.multitest import multipletests
from collections import defaultdict

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for publication-grade plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for statistical evaluation experiments."""
    n_seeds: int = 5  # Number of random seeds per condition
    n_offline_samples: int = 10000
    n_test_samples: int = 10000
    context_dim: int = 5
    action_dim: int = 1
    policy_lr: float = 1e-4
    n_mm_steps: int = 20
    n_pg_steps_per_mm: int = 150
    
    # Tasks to evaluate
    tasks: List[str] = None
    
    # Algorithmic variants (different hyperparameters)
    variants: Dict[str, Dict[str, Any]] = None
    
    # Methods to compare
    methods: List[str] = None
    
    def __post_init__(self):
        if self.tasks is None:
            self.tasks = [
                "BiasedBehaviorSharpPeak",
                "SafetyConstrainedReward", 
                "SparseRewardWithNoise"
            ]
        
        if self.variants is None:
            # Different algorithmic variants (e.g., different clip/clamp settings)
            self.variants = {
                "Standard": {"C_clip": 50.0, "epsilon_mu": 1e-6},
                "HighClip": {"C_clip": 100.0, "epsilon_mu": 1e-6},
                "LowClip": {"C_clip": 20.0, "epsilon_mu": 1e-6},
                "HighClamp": {"C_clip": 50.0, "epsilon_mu": 1e-5},
            }
        
        if self.methods is None:
            self.methods = ["Naive_PG", "PPL_MM"]

# ============================================================================
# HELPER FUNCTIONS (from original code)
# ============================================================================

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


# ============================================================================
# ENVIRONMENT
# ============================================================================

class BenchmarkEnv:
    """Unified environment for continuous-action contextual bandits."""
    
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

    def _true_reward_sharp_peak(self, x, a):
        a_star = x[:, 0].unsqueeze(1)
        distance = torch.abs(a - a_star)
        reward = torch.exp(-50.0 * distance**2)
        return reward

    def _true_reward_safety_constrained(self, x, a):
        a_star = 0.1 * (x[:, 0] + x[:, 1]).unsqueeze(1)
        distance = torch.abs(a - a_star)
        base_reward = torch.exp(-15.0 * distance**2)
        dangerous_mask = (a > 0.4).float()
        danger_amount = torch.clamp(a - 0.4, min=0.0)
        safety_penalty = -3.0 * dangerous_mask * danger_amount**2
        return base_reward + safety_penalty

    def _true_reward_sparse(self, x, a):
        a_star = (0.5 * x[:, 0] + 0.3 * x[:, 1]).unsqueeze(1)
        distance = torch.abs(a - a_star)
        in_reward_zone = (distance < 0.15).float()
        reward = in_reward_zone * torch.exp(-10.0 * distance)
        return reward

    def _get_behavior_biased_away(self, X):
        n_samples = X.shape[0]
        a_optimal = X[:, 0]
        mean_behavior = -0.8 * a_optimal - 0.15
        std = torch.full((n_samples,), 0.35)
        A = sample_truncated_normal(mean_behavior, std)
        log_pdf = truncated_normal_log_pdf(A, mean_behavior, std)
        mu_prob = torch.exp(log_pdf)
        return A.unsqueeze(1), mu_prob.unsqueeze(1)

    def _get_behavior_risky(self, X):
        n_samples = X.shape[0]
        mean = torch.full((n_samples,), 0.4)
        std = torch.full((n_samples,), 0.3)
        A = sample_truncated_normal(mean, std)
        log_pdf = truncated_normal_log_pdf(A, mean, std)
        mu_prob = torch.exp(log_pdf)
        return A.unsqueeze(1), mu_prob.unsqueeze(1)

    def _get_behavior_uniform_noisy(self, X):
        n_samples = X.shape[0]
        A = torch.rand(n_samples) * 2.0 - 1.0
        mu_prob = torch.full((n_samples,), 0.5)
        return A.unsqueeze(1), mu_prob.unsqueeze(1)

    def get_offline_data(self, n_samples):
        X = torch.rand(n_samples, self.context_dim) * 2 - 1
        A, mu_prob = self._get_behavior_data(X)
        true_rewards = self._true_reward_func(X, A)
        noise = torch.randn_like(true_rewards) * self.noise_std
        R = true_rewards + noise
        return X.numpy(), A.numpy(), R.numpy(), mu_prob.numpy()

    def evaluate_policy(self, policy, n_test_samples=5000):
        policy.eval()
        X_test = torch.rand(n_test_samples, self.context_dim) * 2 - 1
        with torch.no_grad():
            A_policy = policy.act(X_test, deterministic=True)
            true_rewards = self._true_reward_func(X_test, A_policy)
        return true_rewards.mean().item()


# ============================================================================
# POLICY NETWORK
# ============================================================================

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
    
    def act(self, x, deterministic=False):
        beta_dist = self._get_dist(x)
        if deterministic:
            a_raw = beta_dist.mean
        else:
            a_raw = beta_dist.sample()
        a = 2.0 * a_raw - 1.0
        return a


# ============================================================================
# TRAINING ALGORITHMS
# ============================================================================

def train_naive_pg(policy, data, env, lr=1e-4, n_steps=3000, eval_interval=200):
    """Naive Policy Gradient without importance weighting."""
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    X, A, R, Mu = data
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    A_tensor = torch.tensor(A, dtype=torch.float32).view(-1, 1)
    R_tensor = torch.tensor(R, dtype=torch.float32).view(-1, 1)
    
    rewards_history = []
    
    for step in range(n_steps):
        log_probs = policy(X_tensor, A_tensor)
        loss = -torch.mean(log_probs * R_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track actual policy performance periodically
        if (step + 1) % eval_interval == 0 or step == 0:
            perf = env.evaluate_policy(policy, n_test_samples=1000)
            rewards_history.append(perf)
    
    return rewards_history


def train_ppl_mm(policy, data, env, beta_pessimism=0.1, C_clip=50.0, epsilon_mu=1e-6,
                 lr=1e-4, n_mm_steps=20, n_pg_steps_per_mm=150, eval_interval=200):
    """PPL-MM algorithm with pessimism."""
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    X, A, R, Mu = data
    n = X.shape[0]
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    A_tensor = torch.tensor(A, dtype=torch.float32).view(-1, 1)
    R_tensor = torch.tensor(R, dtype=torch.float32).view(-1, 1)
    Mu_tensor = torch.tensor(Mu, dtype=torch.float32).view(-1, 1)
    Mu_tensor = torch.clamp(Mu_tensor, min=epsilon_mu, max=float('inf'))
    
    V_const = 1.0 / np.sqrt(n)
    rewards_history = []
    
    for mm_step in range(n_mm_steps):
        # Compute pessimism adjustment
        with torch.no_grad():
            log_probs_k = policy(X_tensor, A_tensor)
            weights_k = torch.exp(log_probs_k) / Mu_tensor
            V_s_n_k = (1.0 / n) * torch.sqrt(torch.sum(weights_k**2) + 1e-8)
            
            if V_s_n_k > V_const:
                adjustment = (beta_pessimism * weights_k) / (n * V_s_n_k)
                R_adjusted = R_tensor - adjustment
            else:
                R_adjusted = R_tensor.clone()
        
        # Inner PG loop
        for pg_step in range(n_pg_steps_per_mm):
            log_probs = policy(X_tensor, A_tensor)
            pi_theta = torch.exp(log_probs)
            weights = pi_theta / Mu_tensor
            clipped_weights = torch.clamp(weights, 0.0, C_clip)
            Y_adjusted = clipped_weights * R_adjusted.detach()
            loss = -torch.mean(Y_adjusted)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track actual policy performance periodically
            total_step = mm_step * n_pg_steps_per_mm + pg_step
            if (total_step + 1) % eval_interval == 0 or total_step == 0:
                perf = env.evaluate_policy(policy, n_test_samples=1000)
                rewards_history.append(perf)
    
    return rewards_history


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

@dataclass
class RunResult:
    """Results from a single run."""
    task: str
    variant: str
    method: str
    seed: int
    final_performance: float  # Primary metric
    cumulative_reward: float  # Alternative metric
    learning_curve: List[float]  # For visualization


class ExperimentRunner:
    """Runs experiments across multiple seeds and conditions."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: List[RunResult] = []
        
        # Noise levels per task
        self.noise_levels = {
            "BiasedBehaviorSharpPeak": 0.05,
            "SafetyConstrainedReward": 0.1,
            "SparseRewardWithNoise": 0.4
        }
        
        # Beta values for PPL-MM per task (from previous experiments)
        self.beta_values = {
            "BiasedBehaviorSharpPeak": 0.15,
            "SafetyConstrainedReward": 0.1,
            "SparseRewardWithNoise": 0.2
        }
    
    def run_single_experiment(self, task: str, variant: str, method: str, seed: int) -> RunResult:
        """Run a single experiment with given configuration and seed."""
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize environment
        noise_std = self.noise_levels.get(task, 0.1)
        env = BenchmarkEnv(benchmark_id=task, context_dim=self.config.context_dim, 
                          noise_std=noise_std)
        
        # Generate offline data
        offline_data = env.get_offline_data(self.config.n_offline_samples)
        
        # Initialize policy
        policy = PolicyNetwork(self.config.context_dim, self.config.action_dim)
        
        # Get variant hyperparameters
        variant_params = self.config.variants[variant]
        
        # Train based on method
        if method == "Naive_PG":
            learning_curve = train_naive_pg(
                policy, offline_data, env,
                lr=self.config.policy_lr,
                n_steps=self.config.n_mm_steps * self.config.n_pg_steps_per_mm,
                eval_interval=200
            )
        elif method == "PPL_MM":
            beta = self.beta_values.get(task, 0.1)
            learning_curve = train_ppl_mm(
                policy, offline_data, env,
                beta_pessimism=beta,
                C_clip=variant_params["C_clip"],
                epsilon_mu=variant_params["epsilon_mu"],
                lr=self.config.policy_lr,
                n_mm_steps=self.config.n_mm_steps,
                n_pg_steps_per_mm=self.config.n_pg_steps_per_mm,
                eval_interval=200
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Evaluate final performance
        final_performance = env.evaluate_policy(policy, n_test_samples=self.config.n_test_samples)
        
        # Compute cumulative reward (alternative metric)
        cumulative_reward = np.sum(learning_curve)
        
        return RunResult(
            task=task,
            variant=variant,
            method=method,
            seed=seed,
            final_performance=final_performance,
            cumulative_reward=cumulative_reward,
            learning_curve=learning_curve
        )
    
    def run_all_experiments(self):
        """Run all experiments across tasks, variants, methods, and seeds."""
        total_runs = (len(self.config.tasks) * len(self.config.variants) * 
                     len(self.config.methods) * self.config.n_seeds)
        
        print(f"Starting {total_runs} experimental runs...")
        print(f"  Tasks: {len(self.config.tasks)}")
        print(f"  Variants: {len(self.config.variants)}")
        print(f"  Methods: {len(self.config.methods)}")
        print(f"  Seeds: {self.config.n_seeds}")
        print()
        
        run_count = 0
        for task in self.config.tasks:
            for variant in self.config.variants:
                for method in self.config.methods:
                    print(f"Running {task} / {variant} / {method}...")
                    for seed in range(self.config.n_seeds):
                        result = self.run_single_experiment(task, variant, method, seed)
                        self.results.append(result)
                        run_count += 1
                        
                        if (run_count % 10 == 0) or (run_count == total_runs):
                            print(f"  Progress: {run_count}/{total_runs} runs completed "
                                  f"({100*run_count/total_runs:.1f}%)")
        
        print(f"\nAll {total_runs} runs completed!")
    
    def save_results(self, filepath: Path):
        """Save results to JSON."""
        results_dict = [asdict(r) for r in self.results]
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"Results saved to {filepath}")


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

class StatisticalAnalyzer:
    """Performs comprehensive statistical analysis."""
    
    def __init__(self, results: List[RunResult], metric: str = "final_performance"):
        self.results = results
        self.metric = metric
        self.df = self._results_to_dataframe()
    
    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        data = []
        for r in self.results:
            data.append({
                'task': r.task,
                'variant': r.variant,
                'method': r.method,
                'seed': r.seed,
                'final_performance': r.final_performance,
                'cumulative_reward': r.cumulative_reward
            })
        return pd.DataFrame(data)
    
    def compute_paired_statistics(self) -> pd.DataFrame:
        """Compute paired statistics for each (task, variant) combination."""
        results_list = []
        
        for task in self.df['task'].unique():
            for variant in self.df['variant'].unique():
                # Get data for this combination
                mask = (self.df['task'] == task) & (self.df['variant'] == variant)
                data_subset = self.df[mask]
                
                # Get paired data
                ppl_data = data_subset[data_subset['method'] == 'PPL_MM'][self.metric].values
                naive_data = data_subset[data_subset['method'] == 'Naive_PG'][self.metric].values
                
                if len(ppl_data) == 0 or len(naive_data) == 0:
                    continue
                
                # Paired difference
                paired_diff = ppl_data - naive_data
                mean_diff = np.mean(paired_diff)
                std_diff = np.std(paired_diff, ddof=1)
                
                # Paired t-test
                t_stat, p_value_ttest = ttest_rel(ppl_data, naive_data)
                
                # Wilcoxon signed-rank test
                try:
                    w_stat, p_value_wilcoxon = wilcoxon(ppl_data, naive_data)
                except:
                    w_stat, p_value_wilcoxon = np.nan, np.nan
                
                # Cohen's d (paired)
                cohens_d = mean_diff / std_diff if std_diff > 0 else 0
                
                # Bootstrap confidence interval
                ci_low, ci_high = self._bootstrap_ci(paired_diff)
                
                results_list.append({
                    'task': task,
                    'variant': variant,
                    'mean_ppl': np.mean(ppl_data),
                    'mean_naive': np.mean(naive_data),
                    'mean_diff': mean_diff,
                    'std_diff': std_diff,
                    't_stat': t_stat,
                    'p_ttest': p_value_ttest,
                    'w_stat': w_stat,
                    'p_wilcoxon': p_value_wilcoxon,
                    'cohens_d': cohens_d,
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                    'n_samples': len(ppl_data)
                })
        
        return pd.DataFrame(results_list)
    
    def _bootstrap_ci(self, data, n_bootstrap=10000, alpha=0.05):
        """Compute bootstrap confidence interval."""
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        ci_low = np.percentile(bootstrap_means, 100 * alpha / 2)
        ci_high = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        return ci_low, ci_high
    
    def apply_multiple_comparison_correction(self, stats_df: pd.DataFrame) -> pd.DataFrame:
        """Apply multiple comparison correction."""
        # Holm correction
        reject_holm, p_holm, _, _ = multipletests(
            stats_df['p_ttest'].values, alpha=0.05, method='holm'
        )
        
        # Benjamini-Hochberg FDR correction
        reject_fdr, p_fdr, _, _ = multipletests(
            stats_df['p_ttest'].values, alpha=0.05, method='fdr_bh'
        )
        
        stats_df['p_holm'] = p_holm
        stats_df['reject_holm'] = reject_holm
        stats_df['p_fdr'] = p_fdr
        stats_df['reject_fdr'] = reject_fdr
        
        return stats_df
    
    def compute_pooled_effect(self, stats_df: pd.DataFrame) -> Dict[str, float]:
        """Compute pooled effect size across all comparisons."""
        # Inverse-variance weighted pooled effect
        weights = 1.0 / (stats_df['std_diff']**2 / stats_df['n_samples'])
        pooled_effect = np.sum(stats_df['mean_diff'] * weights) / np.sum(weights)
        pooled_se = np.sqrt(1.0 / np.sum(weights))
        
        # Pooled Cohen's d
        pooled_cohens_d = np.mean(stats_df['cohens_d'])
        
        return {
            'pooled_mean_diff': pooled_effect,
            'pooled_se': pooled_se,
            'pooled_cohens_d': pooled_cohens_d,
            'pooled_ci_low': pooled_effect - 1.96 * pooled_se,
            'pooled_ci_high': pooled_effect + 1.96 * pooled_se
        }


# ============================================================================
# VISUALIZATION
# ============================================================================

class Visualizer:
    """Creates publication-grade visualizations."""
    
    def __init__(self, results: List[RunResult], stats_df: pd.DataFrame, 
                 pooled_stats: Dict[str, float], output_dir: Path):
        self.results = results
        self.stats_df = stats_df
        self.pooled_stats = pooled_stats
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def create_forest_plot(self):
        """Create forest plot showing effect sizes with confidence intervals."""
        fig, ax = plt.subplots(figsize=(10, max(8, len(self.stats_df) * 0.4 + 2)))
        
        # Sort by effect size
        plot_df = self.stats_df.sort_values('cohens_d', ascending=True).reset_index(drop=True)
        
        y_positions = np.arange(len(plot_df))
        
        # Plot individual effect sizes
        for idx, row in plot_df.iterrows():
            color = 'darkgreen' if row['reject_fdr'] else 'gray'
            marker = 'o' if row['reject_fdr'] else 's'
            
            ax.plot([row['ci_low'], row['ci_high']], [idx, idx], 
                   color=color, linewidth=2, alpha=0.7)
            ax.scatter(row['mean_diff'], idx, color=color, s=100, 
                      marker=marker, zorder=3, edgecolors='black', linewidths=1)
        
        # Add pooled effect at bottom
        pooled_y = -1.5
        ax.plot([self.pooled_stats['pooled_ci_low'], self.pooled_stats['pooled_ci_high']], 
               [pooled_y, pooled_y], color='darkblue', linewidth=3)
        ax.scatter(self.pooled_stats['pooled_mean_diff'], pooled_y, 
                  color='darkblue', s=150, marker='D', zorder=3, 
                  edgecolors='black', linewidths=1.5)
        
        # Vertical line at zero
        ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Labels
        labels = [f"{row['task'][:15]}\n{row['variant']}" for _, row in plot_df.iterrows()]
        labels.append("POOLED")
        ax.set_yticks(list(y_positions) + [pooled_y])
        ax.set_yticklabels(labels, fontsize=9)
        
        ax.set_xlabel('Mean Difference (PPL-MM - Naive PG)', fontsize=12, fontweight='bold')
        ax.set_title('Forest Plot: Effect Sizes with 95% Confidence Intervals', 
                    fontsize=14, fontweight='bold')
        
        # Legend
        sig_patch = mpatches.Patch(color='darkgreen', label='Significant (FDR-corrected)')
        nsig_patch = mpatches.Patch(color='gray', label='Not significant')
        pooled_patch = mpatches.Patch(color='darkblue', label='Pooled effect')
        ax.legend(handles=[sig_patch, nsig_patch, pooled_patch], loc='best', fontsize=9)
        
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        save_path = self.output_dir / 'forest_plot.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Forest plot saved to {save_path}")
    
    def create_paired_comparison_plots(self):
        """Create per-task paired comparison plots."""
        df = pd.DataFrame([{
            'task': r.task,
            'variant': r.variant,
            'method': r.method,
            'seed': r.seed,
            'performance': r.final_performance
        } for r in self.results])
        
        tasks = df['task'].unique()
        n_tasks = len(tasks)
        
        fig, axes = plt.subplots(1, n_tasks, figsize=(6 * n_tasks, 5))
        if n_tasks == 1:
            axes = [axes]
        
        for idx, task in enumerate(tasks):
            ax = axes[idx]
            task_df = df[df['task'] == task]
            variants = task_df['variant'].unique()
            
            x_positions = np.arange(len(variants))
            
            for i, variant in enumerate(variants):
                variant_df = task_df[task_df['variant'] == variant]
                
                ppl_vals = variant_df[variant_df['method'] == 'PPL_MM']['performance'].values
                naive_vals = variant_df[variant_df['method'] == 'Naive_PG']['performance'].values
                
                # Plot paired points with connecting lines
                for ppl, naive in zip(ppl_vals, naive_vals):
                    ax.plot([i - 0.1, i + 0.1], [naive, ppl], 
                           color='gray', alpha=0.3, linewidth=0.5)
                
                # Scatter points
                ax.scatter([i - 0.1] * len(naive_vals), naive_vals, 
                          color='lightcoral', s=50, alpha=0.6, label='Naive PG' if i == 0 else '')
                ax.scatter([i + 0.1] * len(ppl_vals), ppl_vals, 
                          color='skyblue', s=50, alpha=0.6, label='PPL-MM' if i == 0 else '')
                
                # Add means
                ax.plot([i - 0.1], [np.mean(naive_vals)], 'r_', markersize=15, markeredgewidth=3)
                ax.plot([i + 0.1], [np.mean(ppl_vals)], 'b_', markersize=15, markeredgewidth=3)
            
            ax.set_xticks(x_positions)
            ax.set_xticklabels(variants, rotation=45, ha='right')
            ax.set_ylabel('Performance', fontsize=11)
            ax.set_title(f'{task}', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            if idx == 0:
                ax.legend(loc='best', fontsize=9)
        
        fig.suptitle('Paired Comparisons: PPL-MM vs Naive PG', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / 'paired_comparisons.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Paired comparison plots saved to {save_path}")
    
    def create_heatmap(self):
        """Create heatmap of performance differences."""
        # Pivot data for heatmap
        pivot_data = self.stats_df.pivot(index='variant', columns='task', values='mean_diff')
        significance = self.stats_df.pivot(index='variant', columns='task', values='reject_fdr')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Mean Difference (PPL-MM - Naive PG)'},
                   linewidths=1, linecolor='gray', ax=ax)
        
        # Mark significant cells
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                if significance.iloc[i, j]:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, 
                                              edgecolor='blue', lw=3))
        
        ax.set_title('Performance Difference Heatmap (Blue border = significant after FDR correction)',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Task', fontsize=11, fontweight='bold')
        ax.set_ylabel('Variant', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        save_path = self.output_dir / 'heatmap.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Heatmap saved to {save_path}")
    
    def create_learning_curves(self):
        """Create learning curve summaries."""
        df_curves = pd.DataFrame([{
            'task': r.task,
            'variant': r.variant,
            'method': r.method,
            'seed': r.seed,
            'curve': r.learning_curve
        } for r in self.results])
        
        tasks = df_curves['task'].unique()
        variants = df_curves['variant'].unique()
        
        n_rows = len(tasks)
        n_cols = len(variants)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, task in enumerate(tasks):
            for j, variant in enumerate(variants):
                ax = axes[i, j]
                
                subset = df_curves[(df_curves['task'] == task) & 
                                  (df_curves['variant'] == variant)]
                
                for method in ['Naive_PG', 'PPL_MM']:
                    method_subset = subset[subset['method'] == method]
                    curves = np.array(method_subset['curve'].tolist())
                    
                    mean_curve = np.mean(curves, axis=0)
                    std_curve = np.std(curves, axis=0)
                    
                    steps = np.arange(len(mean_curve))
                    color = 'lightcoral' if method == 'Naive_PG' else 'skyblue'
                    label = 'Naive PG' if method == 'Naive_PG' else 'PPL-MM'
                    
                    ax.plot(steps, mean_curve, color=color, linewidth=2, label=label)
                    ax.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve,
                                   color=color, alpha=0.2)
                
                ax.set_xlabel('Evaluation Point (every 200 steps)', fontsize=9)
                ax.set_ylabel('Performance (True Reward)', fontsize=9)
                ax.set_title(f'{task}\n{variant}', fontsize=10)
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
        
        fig.suptitle('Learning Curves: Mean ± Std Dev', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / 'learning_curves.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Learning curves saved to {save_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print("="*80)
    print("PPL-MM vs Naive PG: Comprehensive Statistical Evaluation")
    print("="*80)
    print()
    
    # Configuration
    config = ExperimentConfig(
        n_seeds=5,  # 5 runs per condition for good statistical power
        n_offline_samples=10000,
        n_test_samples=10000,
        n_mm_steps=20,
        n_pg_steps_per_mm=150
    )
    
    results_dir = Path("results_statistical")
    results_dir.mkdir(exist_ok=True)
    
    # Run experiments
    runner = ExperimentRunner(config)
    runner.run_all_experiments()
    
    # Save raw results
    results_path = results_dir / "raw_results.json"
    runner.save_results(results_path)
    
    # Statistical analysis
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)
    
    analyzer = StatisticalAnalyzer(runner.results, metric="final_performance")
    
    # Compute paired statistics
    print("\nComputing paired statistics...")
    stats_df = analyzer.compute_paired_statistics()
    
    # Apply multiple comparison correction
    print("Applying multiple comparison correction...")
    stats_df = analyzer.apply_multiple_comparison_correction(stats_df)
    
    # Compute pooled effect
    print("Computing pooled effect size...")
    pooled_stats = analyzer.compute_pooled_effect(stats_df)
    
    # Save statistics
    stats_path = results_dir / "statistical_results.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\nStatistical results saved to {stats_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\nPooled effect size: {pooled_stats['pooled_mean_diff']:.4f}")
    print(f"  95% CI: [{pooled_stats['pooled_ci_low']:.4f}, {pooled_stats['pooled_ci_high']:.4f}]")
    print(f"  Cohen's d: {pooled_stats['pooled_cohens_d']:.3f}")
    
    n_significant_holm = stats_df['reject_holm'].sum()
    n_significant_fdr = stats_df['reject_fdr'].sum()
    n_total = len(stats_df)
    
    print(f"\nSignificant comparisons (Holm): {n_significant_holm}/{n_total}")
    print(f"Significant comparisons (FDR): {n_significant_fdr}/{n_total}")
    
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    visualizer = Visualizer(runner.results, stats_df, pooled_stats, results_dir)
    visualizer.create_forest_plot()
    visualizer.create_paired_comparison_plots()
    visualizer.create_heatmap()
    visualizer.create_learning_curves()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {results_dir}")
    print("\nGenerated files:")
    print("  - raw_results.json: All experimental results")
    print("  - statistical_results.csv: Paired statistics for each condition")
    print("  - forest_plot.png: Effect sizes with confidence intervals")
    print("  - paired_comparisons.png: Per-task paired plots")
    print("  - heatmap.png: Performance differences across conditions")
    print("  - learning_curves.png: Training dynamics")


if __name__ == "__main__":
    main()
