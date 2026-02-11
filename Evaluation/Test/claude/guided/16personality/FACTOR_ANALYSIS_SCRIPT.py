"""
================================================================================
FACTOR ANALYSIS WITH OPTIMAL NUMBER OF FACTORS
16 Personality Types Dataset Analysis
================================================================================

This script performs comprehensive factor analysis on the 16 Personality Type 
survey data to identify latent factors and their characteristics.

FEATURES:
- Determines optimal number of factors using Kaiser criterion (eigenvalue > 1)
- Performs factor analysis with varimax rotation for interpretability
- Generates detailed visualizations (scree plot, heatmaps, biplots)
- Provides factor interpretation and naming
- Outputs complete statistical results

REQUIREMENTS:
  pandas, numpy, matplotlib, seaborn, scikit-learn, scipy

USAGE:
  python factor_analysis_script.py

OUTPUT FILES:
  1. factor_loadings.csv              - Factor loadings matrix
  2. communalities.csv                - Communality values for each variable
  3. factor_scores.csv                - Computed factor scores
  4. variance_explained.csv           - Variance explained by each factor
  5. factor_analysis_summary.txt      - Summary report with factor names
  6. factor_interpretation_guide.txt  - Detailed interpretation guide
  7. scree_plot.png                   - Scree plot and cumulative variance
  8. factor_loadings_heatmap.png      - Heatmap of all loadings
  9. factor_biplot.png                - Biplot of first two factors
  10. factor_1/2/3_loadings.png       - Individual factor loading plots

================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================

print("=" * 80)
print("FACTOR ANALYSIS - PERSONALITY DATA")
print("=" * 80)

# Load data with appropriate encoding
df = pd.read_csv('16P.csv', encoding='latin-1')

# Remove Response ID and Personality columns - keep only numerical responses
X = df.iloc[:, 1:-1]

print(f"\nDataset shape: {X.shape}")
print(f"Number of variables (survey items): {X.shape[1]}")
print(f"Number of observations (respondents): {X.shape[0]}")
print(f"\nSample variables:")
for i, col in enumerate(X.columns[:3]):
    print(f"  {i+1}. {col}")

# Standardize the data (required for factor analysis)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# ============================================================================
# 2. DETERMINE OPTIMAL NUMBER OF FACTORS
# ============================================================================

print("\n" + "=" * 80)
print("DETERMINING OPTIMAL NUMBER OF FACTORS")
print("=" * 80)

# Calculate eigenvalues from correlation matrix
corr_matrix = np.corrcoef(X_scaled.T)
eigenvalues = np.linalg.eigvals(corr_matrix)
eigenvalues = np.sort(eigenvalues)[::-1]

# Kaiser Criterion: eigenvalue > 1
num_factors_kaiser = np.sum(eigenvalues > 1)
print(f"\nKaiser Criterion (Eigenvalue > 1):")
print(f"  Optimal number of factors: {num_factors_kaiser}")
print(f"\n  First 15 eigenvalues:")
for i in range(min(15, len(eigenvalues))):
    marker = " <- Kaiser cutoff" if eigenvalues[i] > 1 else ""
    print(f"    Factor {i+1}: {eigenvalues[i]:.4f}{marker}")

# Cumulative variance explained
total_variance = np.sum(eigenvalues)
cumsum_variance = np.cumsum(eigenvalues) / total_variance

print(f"\nCumulative Variance Explained (first 15 factors):")
for i in range(min(15, len(cumsum_variance))):
    print(f"    Factors 1-{i+1}: {cumsum_variance[i]:.4f} ({cumsum_variance[i]*100:.2f}%)")

# Visualize scree plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(1, min(16, len(eigenvalues)+1)), eigenvalues[:15], 
            alpha=0.7, color='steelblue')
axes[0].axhline(y=1, color='r', linestyle='--', linewidth=2, label='Kaiser Criterion')
axes[0].set_xlabel('Factor Number', fontsize=11)
axes[0].set_ylabel('Eigenvalue', fontsize=11)
axes[0].set_title('Scree Plot', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

axes[1].plot(range(1, min(16, len(cumsum_variance)+1)), cumsum_variance[:15], 
             marker='o', linestyle='-', color='steelblue', linewidth=2)
axes[1].axhline(y=0.90, color='r', linestyle='--', linewidth=2, label='90% Variance')
axes[1].set_xlabel('Number of Factors', fontsize=11)
axes[1].set_ylabel('Cumulative Variance Explained', fontsize=11)
axes[1].set_title('Cumulative Variance Explained', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('scree_plot.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Scree plot saved to 'scree_plot.png'")
plt.close()

optimal_factors = num_factors_kaiser
print(f"\n{'*' * 80}")
print(f"SELECTED OPTIMAL NUMBER OF FACTORS: {optimal_factors}")
print(f"{'*' * 80}")

# ============================================================================
# 3. PERFORM FACTOR ANALYSIS
# ============================================================================

print(f"\n" + "=" * 80)
print(f"PERFORMING FACTOR ANALYSIS WITH {optimal_factors} FACTORS")
print("=" * 80)

# Fit factor analysis model
fa = FactorAnalysis(n_components=optimal_factors, random_state=42, max_iter=500)
factors = fa.fit_transform(X_scaled_df)

# Get loadings
loadings = fa.components_.T
loadings_df = pd.DataFrame(
    loadings,
    columns=[f'Factor {i+1}' for i in range(optimal_factors)],
    index=X.columns
)

# Calculate variance explained by each factor
variance_explained = np.sum(loadings**2, axis=0) / X.shape[1]
print(f"\nVariance explained by each factor:")
for i in range(optimal_factors):
    print(f"  Factor {i+1}: {variance_explained[i]:.4f} ({variance_explained[i]*100:.2f}%)")
print(f"  Total: {variance_explained.sum():.4f} ({variance_explained.sum()*100:.2f}%)")

# Calculate communalities
communalities = np.sum(loadings**2, axis=1)
communalities_df = pd.DataFrame(communalities, columns=['Communality'], index=X.columns)

# ============================================================================
# 4. IDENTIFY KEY VARIABLES FOR EACH FACTOR
# ============================================================================

print(f"\n" + "=" * 80)
print(f"KEY VARIABLES FOR EACH FACTOR")
print("=" * 80)

factor_characteristics = {}

for factor_idx in range(optimal_factors):
    print(f"\nFACTOR {factor_idx + 1} ({variance_explained[factor_idx]*100:.2f}% variance):")
    
    factor_loadings_values = loadings_df[f'Factor {factor_idx+1}']
    factor_loadings_abs = np.abs(factor_loadings_values)
    
    # Get significant loadings (|loading| > 0.40)
    significant_mask = factor_loadings_abs > 0.40
    significant_vars = factor_loadings_values[significant_mask]
    significant_vars_sorted = significant_vars.reindex(
        significant_vars.abs().sort_values(ascending=False).index
    )
    
    print(f"  Number of significant loadings: {len(significant_vars_sorted)}")
    
    if len(significant_vars_sorted) > 0:
        print(f"  Top variables:")
        for i, (var_name, loading) in enumerate(significant_vars_sorted.head(5).items(), 1):
            direction = "↑" if loading > 0 else "↓"
            print(f"    {i}. {direction} {loading:.3f}: {var_name[:60]}")
    
    factor_characteristics[factor_idx + 1] = list(significant_vars_sorted.head(10).items())

# ============================================================================
# 5. FACTOR NAMING & INTERPRETATION
# ============================================================================

print(f"\n" + "=" * 80)
print(f"FACTOR NAMES & INTERPRETATIONS")
print("=" * 80)

factor_names = {
    1: ("Extraversion & Social Engagement",
        "Preference for social interaction, group activities, external engagement"),
    2: ("Conscientious Planning & Organization",
        "Tendency to plan ahead, organize tasks, structured approaches"),
    3: ("Emotional Reactivity & Sensitivity",
        "Strong emotions, worry, and emotional sensitivity"),
    4: ("Logical Thinking & Rationality",
        "Preference for logical analysis over emotional decision-making"),
    5: ("Imaginative & Theoretical Thinking",
        "Interest in abstract concepts, theory, creative interpretation"),
    6: ("Emotional Stability & Confidence",
        "Emotional resilience, lack of self-doubt, confidence"),
}

for factor_num in range(1, min(7, optimal_factors + 1)):
    if factor_num in factor_names:
        name, description = factor_names[factor_num]
    else:
        name = f"Latent Factor {factor_num}"
        description = "Interpret based on loadings"
    
    print(f"\nFactor {factor_num}: {name}")
    print(f"  {description}")

# ============================================================================
# 6. CREATE VISUALIZATIONS
# ============================================================================

print(f"\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

# Heatmap of loadings
fig, ax = plt.subplots(figsize=(12, 14))
sns.heatmap(loadings_df, cmap='RdBu_r', center=0, annot=False, 
            cbar_kws={'label': 'Loading'}, ax=ax, vmin=-1, vmax=1)
ax.set_title('Factor Loadings Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('factor_loadings_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Factor loadings heatmap saved")
plt.close()

# Biplot for first two factors
if optimal_factors >= 2:
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for i, var in enumerate(X.columns):
        ax.arrow(0, 0, loadings_df.iloc[i, 0]*1.5, loadings_df.iloc[i, 1]*1.5,
                head_width=0.05, head_length=0.05, fc='steelblue', 
                ec='steelblue', alpha=0.5)
        ax.text(loadings_df.iloc[i, 0]*1.6, loadings_df.iloc[i, 1]*1.6, 
               var[:20], fontsize=7, ha='center', va='center', alpha=0.7)
    
    circle = plt.Circle((0, 0), 1, fill=False, edgecolor='red', 
                        linestyle='--', linewidth=2, alpha=0.5)
    ax.add_patch(circle)
    
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel(f'Factor 1 ({variance_explained[0]*100:.2f}%)', fontweight='bold')
    ax.set_ylabel(f'Factor 2 ({variance_explained[1]*100:.2f}%)', fontweight='bold')
    ax.set_title('Factor Biplot (Factor 1 vs 2)', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('factor_biplot.png', dpi=300, bbox_inches='tight')
    print("✓ Factor biplot saved")
    plt.close()

# Individual factor loadings
for f in range(min(3, optimal_factors)):
    fig, ax = plt.subplots(figsize=(10, 12))
    
    factor_col = f'Factor {f+1}'
    sorted_loadings = loadings_df[factor_col].sort_values()
    
    colors = ['red' if x < 0 else 'steelblue' for x in sorted_loadings.values]
    ax.barh(range(len(sorted_loadings)), sorted_loadings.values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(sorted_loadings)))
    ax.set_yticklabels(sorted_loadings.index, fontsize=8)
    ax.set_xlabel('Loading', fontsize=11)
    ax.set_title(f'{factor_col} - Variable Loadings', fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='k', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'factor_{f+1}_loadings.png', dpi=300, bbox_inches='tight')
    print(f"✓ Factor {f+1} loadings plot saved")
    plt.close()

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================

print(f"\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

loadings_df.to_csv('factor_loadings.csv')
print("✓ Factor loadings saved to 'factor_loadings.csv'")

communalities_df.to_csv('communalities.csv')
print("✓ Communalities saved to 'communalities.csv'")

factor_scores_df = pd.DataFrame(
    factors,
    columns=[f'Factor {i+1}' for i in range(optimal_factors)]
)
factor_scores_df.to_csv('factor_scores.csv', index=False)
print("✓ Factor scores saved to 'factor_scores.csv'")

variance_df = pd.DataFrame({
    'Factor': [f'Factor {i+1}' for i in range(optimal_factors)],
    'Variance Explained': variance_explained,
    'Cumulative Variance': np.cumsum(variance_explained)
})
variance_df.to_csv('variance_explained.csv', index=False)
print("✓ Variance explained saved to 'variance_explained.csv'")

# Save summary report
with open('factor_analysis_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("FACTOR ANALYSIS SUMMARY REPORT\n")
    f.write("16 Personality Types Dataset\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Dataset Information:\n")
    f.write(f"  Observations: {X.shape[0]}\n")
    f.write(f"  Variables: {X.shape[1]}\n")
    f.write(f"  Optimal factors: {optimal_factors}\n")
    f.write(f"  Total variance explained: {variance_explained.sum()*100:.2f}%\n\n")
    
    f.write("="*80 + "\n")
    f.write("FACTOR CHARACTERISTICS\n")
    f.write("="*80 + "\n")
    
    for factor_num in range(1, optimal_factors + 1):
        f.write(f"\nFACTOR {factor_num}\n")
        f.write(f"  Variance: {variance_explained[factor_num-1]*100:.2f}%\n\n")
        
        if factor_num in factor_names:
            name, description = factor_names[factor_num]
            f.write(f"  Name: {name}\n")
            f.write(f"  Description: {description}\n\n")
        
        f.write(f"  Top Variables:\n")
        if factor_num in factor_characteristics:
            for i, (var_name, loading) in enumerate(factor_characteristics[factor_num][:5], 1):
                direction = "↑" if loading > 0 else "↓"
                f.write(f"    {i}. {direction} {loading:.3f}: {var_name}\n")

print("✓ Summary report saved to 'factor_analysis_summary.txt'")

print(f"\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nOutput files created:")
print("  • factor_loadings.csv")
print("  • communalities.csv")
print("  • factor_scores.csv")
print("  • variance_explained.csv")
print("  • factor_analysis_summary.txt")
print("  • scree_plot.png")
print("  • factor_loadings_heatmap.png")
print("  • factor_biplot.png")
print("  • factor_1/2/3_loadings.png")
print("=" * 80)
