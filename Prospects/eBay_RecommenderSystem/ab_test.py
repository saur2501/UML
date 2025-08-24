"""
A/B Testing Demonstration
A short program demonstrating A/B testing concepts and statistical analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

class ABTest:
    """Simple A/B testing class."""

    def __init__(self, control_data, treatment_data):
        self.control_data = np.array(control_data)
        self.treatment_data = np.array(treatment_data)

    def calculate_metrics(self):
        """Calculate basic metrics for both groups."""
        control_mean = np.mean(self.control_data)
        treatment_mean = np.mean(self.treatment_data)
        control_std = np.std(self.control_data)
        treatment_std = np.std(self.treatment_data)

        return {
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'control_std': control_std,
            'treatment_std': treatment_std,
            'lift': (treatment_mean - control_mean) / control_mean * 100
        }

    def t_test(self):
        """Perform independent t-test."""
        t_stat, p_value = stats.ttest_ind(self.control_data, self.treatment_data)
        return {'t_statistic': t_stat, 'p_value': p_value}

    def chi_square_test(self, control_conversions, control_total,
                        treatment_conversions, treatment_total):
        """Perform chi-square test for conversion rates."""
        # Create contingency table
        control_non_conversions = control_total - control_conversions
        treatment_non_conversions = treatment_total - treatment_conversions

        contingency_table = [
            [control_conversions, control_non_conversions],
            [treatment_conversions, treatment_non_conversions]
        ]

        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

        control_rate = control_conversions / control_total
        treatment_rate = treatment_conversions / treatment_total
        lift = (treatment_rate - control_rate) / control_rate * 100

        return {
            'chi2_statistic': chi2,
            'p_value': p_value,
            'control_rate': control_rate,
            'treatment_rate': treatment_rate,
            'lift': lift
        }

def demonstrate_ab_testing():
    """Demonstrate A/B testing with examples."""
    print("A/B TESTING DEMONSTRATION")
    print("=" * 40)

    # Example 1: Continuous data (revenue per user)
    print("1. Continuous Data Test (Revenue per User)")
    print("-" * 40)

    # Generate sample data
    np.random.seed(42)
    control_revenue = np.random.normal(100, 15, 1000)  # Control group
    treatment_revenue = np.random.normal(105, 15, 1000)  # Treatment group

    ab_test = ABTest(control_revenue, treatment_revenue)
    metrics = ab_test.calculate_metrics()
    t_results = ab_test.t_test()

    print(f"Control Mean: ${metrics['control_mean']:.2f}")
    print(f"Treatment Mean: ${metrics['treatment_mean']:.2f}")
    print(f"Lift: {metrics['lift']:.2f}%")
    print(f"T-statistic: {t_results['t_statistic']:.3f}")
    print(f"P-value: {t_results['p_value']:.4f}")

    if t_results['p_value'] < 0.05:
        print("Result: Statistically significant (p < 0.05)")
    else:
        print("Result: Not statistically significant (p >= 0.05)")

    print()

    # Example 2: Conversion rate test
    print("2. Conversion Rate Test")
    print("-" * 40)

    control_conversions = 150
    control_total = 1000
    treatment_conversions = 180
    treatment_total = 1000

    chi_results = ab_test.chi_square_test(control_conversions, control_total,
                                       treatment_conversions, treatment_total)

    print(f"Control Conversion Rate: {chi_results['control_rate']:.2%}")
    print(f"Treatment Conversion Rate: {chi_results['treatment_rate']:.2%}")
    print(f"Lift: {chi_results['lift']:.2f}%")
    print(f"Chi2 Statistic: {chi_results['chi2_statistic']:.3f}")
    print(f"P-value: {chi_results['p_value']:.4f}")

    if chi_results['p_value'] < 0.05:
        print("Result: Statistically significant (p < 0.05)")
    else:
        print("Result: Not statistically significant (p >= 0.05)")

    print()

    # Example 3: Sample size calculation
    print("3. Sample Size Calculation")
    print("-" * 40)

    def calculate_sample_size(baseline_rate, minimum_detectable_effect):
        """Simple sample size calculation for A/B test."""
        # Simplified formula for sample size calculation
        p1 = baseline_rate
        p2 = baseline_rate + minimum_detectable_effect
        p_avg = (p1 + p2) / 2

        # Z-scores for 95% confidence and 80% power
        z_alpha = 1.96  # Two-tailed test
        z_beta = 0.84   # 80% power

        # Sample size per group
        n_per_group = ((z_alpha * np.sqrt(2 * p_avg * (1 - p_avg)) +
                       z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / \
                      (minimum_detectable_effect ** 2)

        return int(np.ceil(n_per_group * 2))  # Total for both groups

    baseline_rate = 0.10  # 10% baseline conversion
    minimum_detectable_effect = 0.02  # 2% absolute improvement

    required_sample_size = calculate_sample_size(baseline_rate, minimum_detectable_effect)

    print(f"Baseline Conversion Rate: {baseline_rate:.1%}")
    print(f"Minimum Detectable Effect: {minimum_detectable_effect:.1%}")
    print(f"Significance Level: 5%")
    print(f"Power: 80%")
    print(f"Required Sample Size: {required_sample_size} ({required_sample_size//2} per group)")

def plot_ab_results():
    """Plot A/B test results."""
    print("\n4. Visualization")
    print("-" * 40)

    # Generate data for visualization
    np.random.seed(42)
    control_data = np.random.normal(100, 15, 1000)
    treatment_data = np.random.normal(105, 15, 1000)

    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(control_data, bins=30, alpha=0.7, label='Control', color='blue')
    plt.hist(treatment_data, bins=30, alpha=0.7, label='Treatment', color='red')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('A/B Test Results Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('ab_test_results.png', dpi=300, bbox_inches='tight')
    print("A/B test results plot saved as: ab_test_results.png")
    plt.close()

def main():
    """Main function."""
    demonstrate_ab_testing()
    plot_ab_results()

    print("\n" + "=" * 40)
    print("KEY CONCEPTS")
    print("-" * 40)
    print("1. Null Hypothesis: No difference between groups")
    print("2. P-value: Probability of observing results by chance")
    print("3. Significance Level: Threshold for statistical significance (usually 5%)")
    print("4. Power: Probability of detecting true effect (usually 80%)")
    print("5. Sample Size: Ensure enough data to detect meaningful effects")
    print("6. Lift: Relative improvement of treatment over control")

if __name__ == "__main__":
    main()
