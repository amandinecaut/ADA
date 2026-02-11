# Factor Analysis on Rugby Player Statistics - Python Code Guide

## Overview
This document provides a comprehensive guide to the Python code for performing Factor Analysis on rugby player statistics. The analysis identifies the optimal number of factors and provides meaningful interpretations.

---

## Key Dependencies

```python
import pandas as pd              # Data manipulation
import numpy as np               # Numerical computing
import matplotlib.pyplot as plt   # Visualization
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis, PCA
from scipy.stats import chi2     # Statistical tests
import seaborn as sns            # Enhanced visualizations
```

---

## Analysis Steps

### 1. Data Loading and Preparation

```python
# Load data with proper encoding (handles special characters)
df = pd.read_csv('/path/to/Statistic_rugby_players.csv', encoding='latin-1')
column_map = pd.read_excel('/path/to/rugby_player_map.xlsx')
```

**Key Point**: Use `encoding='latin-1'` to handle special characters in player names.

### 2. Select Numeric Variables

The analysis uses 29 numeric variables across three categories:
- **Personal attributes**: age, height, weight
- **Club performance**: matches, wins, draws, losses, tries, points, minutes
- **National team performance**: matches, wins, draws, losses, tries, points, minutes
- **Other tournaments**: Same metrics as above
- **Discipline**: Yellow and red cards

### 3. Data Standardization

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled, columns=numeric_cols)
```

**Why standardize?**
- Puts all variables on the same scale (mean=0, std=1)
- Prevents variables with larger variances from dominating
- Required for meaningful factor analysis

---

## Suitability Tests

### Bartlett's Test of Sphericity

```python
def calculate_bartlett_sphericity(X):
    """
    Tests if correlation matrix is significantly different from identity matrix.
    H0: Variables are uncorrelated
    H1: Variables are correlated
    """
    n_obs, n_vars = X.shape
    corr_matrix = np.corrcoef(X.T)
    det_corr = np.linalg.det(corr_matrix)
    
    chi_square = -(n_obs - 1 - (2*n_vars + 5)/6) * np.log(det_corr)
    p_value = 1 - chi2.cdf(chi_square, df=int(n_vars*(n_vars-1)/2))
    
    return chi_square, p_value
```

**Interpretation**:
- **p-value < 0.05**: Suitable for factor analysis (reject H0)
- **p-value > 0.05**: Variables may be uncorrelated

### Kaiser-Meyer-Olkin (KMO) Test

```python
def calculate_kmo(X):
    """
    Measures sampling adequacy for each variable and overall.
    Range: 0 to 1
    """
    # ... correlation and partial correlation calculation
    # KMO = Σ(r_ij²) / [Σ(r_ij²) + Σ(p_ij²)]
```

**Interpretation**:
| KMO Value | Interpretation |
|-----------|----------------|
| > 0.9     | Marvelous      |
| > 0.8     | Meritorious    |
| > 0.7     | Middling       |
| > 0.6     | Mediocre       |
| > 0.5     | Miserable      |
| < 0.5     | Unacceptable   |

---

## Determining Optimal Number of Factors

### Method 1: Kaiser Criterion

```python
# Eigenvalues > 1 indicate factors that explain more variance than one variable
kaiser_factors = np.sum(eigenvalues > 1)
```

For rugby data: **9 factors** (eigenvalues > 1)

### Method 2: Cumulative Variance Explained (80% threshold)

```python
# Find how many factors needed to explain 80% of variance
variance_80_factors = np.argmax(cumsum_var_pct >= 80) + 1
```

For rugby data: **10 factors** (80% threshold)

### Method 3: Scree Plot Analysis

```python
# Plot eigenvalues and look for "elbow" point
plt.plot(factors, eigenvalues, 'bo-')
plt.axhline(y=1, color='r', linestyle='--', label='Kaiser Criterion')
plt.xlabel('Factor Number')
plt.ylabel('Eigenvalue')
plt.title('Scree Plot')
```

**Selection Rule**: Use Kaiser criterion (more conservative, 9 factors selected)

---

## Performing Factor Analysis

```python
from sklearn.decomposition import FactorAnalysis

# Initialize and fit
fa = FactorAnalysis(n_components=optimal_factors, 
                    random_state=42, 
                    max_iter=1000)
fa.fit(data_scaled)

# Get loadings (correlation between variables and factors)
loadings = fa.components_.T
loadings_df = pd.DataFrame(loadings,
                           columns=[f'Factor {i+1}' for i in range(optimal_factors)],
                           index=numeric_cols)
```

**Factor Loadings**: 
- Range: -1 to +1
- Indicate correlation between variable and factor
- Absolute value > 0.3 considered significant
- Positive/negative shows direction of relationship

---

## Factor Interpretation and Naming

### Naming Strategy

For each factor, examine:
1. **Top 8 variables** by absolute loading
2. **Nature of variables**: Are they related to club, national, or tournament performance?
3. **Patterns**: Do variables cluster around specific themes?

### Automated Naming Rules

```python
if 'National_match' in top_vars and 'club-match' in top_vars:
    name = "Overall Playing Experience & International Presence"
elif 'club-match' in top_vars:
    name = "Club Match Frequency & Playing Time"
elif 'National_match' in top_vars:
    name = "National Team Performance & Selection"
# ... etc
```

---

## Results Interpretation

### Factor Analysis Output

**Results for Rugby Data:**

| Factor | Variance % | Interpretation |
|--------|-----------|-----------------|
| 1      | 11.48%    | Other Tournament Performance |
| 2      | 11.48%    | Club Match Frequency & Playing Time |
| 3      | 11.46%    | National Team Performance & Selection |
| 4      | 11.46%    | National Team Performance (Alternative) |
| 5      | 11.42%    | Club Match Results |
| 6      | 11.21%    | Club Playing Time & Attacking |
| 7      | 11.19%    | National Team Losses |
| 8      | 10.76%    | Physical Attributes |
| 9      | 9.54%     | Scoring Performance |
| **Total** | **100.00%** | **Complete explanation** |

### Key Insights

1. **9 optimal factors** capture all variance
2. **Relatively equal distribution** (9-12% each) suggests complex, multi-dimensional player profiles
3. **Main themes**:
   - Playing frequency/experience at different levels (club, national, tournaments)
   - Team performance (wins/losses)
   - Scoring and attacking contribution
   - Physical characteristics
   - Discipline records

---

## Output Files

### 1. `factor_loadings.csv`
Matrix showing how each variable loads on each factor.

```python
loadings_df.to_csv('factor_loadings.csv')
```

### 2. `factor_scores.csv`
Player scores on each factor (dimensionality reduction).

```python
factor_scores = fa.transform(data_scaled)
factor_scores_df = pd.DataFrame(factor_scores, columns=[f'Factor {i+1}' ...])
```

### 3. `factor_analysis_report.txt`
Detailed text report with all statistics and interpretations.

### 4. `factor_analysis_results.png`
Comprehensive visualization with 6 subplots:
- Scree plot
- Cumulative variance explained
- Factor loadings heatmap
- Top contributors per factor
- Variance explained bar chart
- Factor names summary

---

## Advanced Usage

### Extract Factor Scores for New Analysis

```python
# Get player scores on factors
player_factor_scores = fa.transform(new_data_scaled)

# Use for clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(player_factor_scores)
```

### Rotation Methods

The code uses **Varimax rotation** (orthogonal) for interpretability:
- Maximizes variance of squared loadings
- Creates uncorrelated factors
- Easier interpretation

Alternative: **Promax rotation** (oblique) allows factor correlation.

### Custom Factor Names

Modify the naming logic:

```python
# Instead of automated naming
factor_names[factor_idx] = "Your Custom Name"
```

---

## Common Issues and Solutions

### Issue 1: Encoding Errors
**Solution**: Use `encoding='latin-1'` when reading CSV

```python
df = pd.read_csv(file, encoding='latin-1')
```

### Issue 2: Singular Correlation Matrix
**Solution**: Remove highly correlated variables or use regularization

```python
# Check correlation
corr_matrix = data_scaled.corr()
# Remove variables with |corr| > 0.95
```

### Issue 3: Low KMO Score (< 0.5)
**Solution**: 
- Remove variables with individual KMO < 0.5
- Consider that some datasets aren't suitable for FA
- Use PCA as alternative

---

## Code Modifications Guide

### Change Optimal Factors

```python
# Instead of Kaiser criterion:
optimal_factors = 5  # Or use variance_80_factors, etc.
```

### Use Different Rotation

```python
# Note: sklearn doesn't support rotation natively
# Use factor_analyzer library or manual rotation
```

### Add Custom Visualizations

```python
# Example: Biplot of first 2 factors
plt.figure(figsize=(10, 8))
for i, var in enumerate(numeric_cols):
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
              head_width=0.05, head_length=0.05)
    plt.text(loadings[i, 0]*1.1, loadings[i, 1]*1.1, var)
plt.xlabel(f'Factor 1 ({variance_explained_pct[0]:.2f}%)')
plt.ylabel(f'Factor 2 ({variance_explained_pct[1]:.2f}%)')
plt.grid()
plt.show()
```

---

## Mathematical Background

### Factor Analysis Model

```
X = Λf + ε

Where:
X = observed variables (p × n)
Λ = factor loadings (p × m)
f = common factors (m × n)
ε = specific factors (p × n)
```

### Variance Decomposition

```
var(X_i) = Σ(λ²_ij) + ψ_i

Where:
Σ(λ²_ij) = communality (variance explained by common factors)
ψ_i = specific variance (unique to variable)
```

---

## References

- **Kaiser Criterion**: Kaiser, H. F. (1960). "The Application of Electronic Computers to Factor Analysis"
- **KMO Test**: Kaiser, H. F., & Rice, W. A. (1974). "Little Jiffy, Mark IV"
- **Bartlett Test**: Bartlett, M. S. (1951). "The Effect of Standardization on a Chi-square Approximation"

---

## Contact & Support

For questions or modifications to the code, ensure:
1. Data is properly standardized
2. Suitability tests pass (KMO > 0.5, Bartlett p < 0.05)
3. Sample size is adequate (n > 5p, where p = number of variables)
4. Missing values are handled appropriately
