# Cluster Analysis on Extracted Factors - Python Code Guide

## Overview
This comprehensive guide covers the Python code for performing K-means clustering on the extracted rugby player factors, with automatic cluster naming and three-sentence descriptions for each cluster.

---

## Key Dependencies

```python
import pandas as pd              # Data manipulation
import numpy as np               # Numerical computing
import matplotlib.pyplot as plt   # Visualization
import seaborn as sns            # Enhanced plots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
```

---

## Main Analysis Steps

### Step 1: Load Factor Scores

```python
factor_scores_df = pd.read_csv('/mnt/user-data/outputs/factor_scores.csv')

players = factor_scores_df['Player'].values
factor_columns = [col for col in factor_scores_df.columns if col.startswith('Factor')]
factor_data = factor_scores_df[factor_columns].values
```

**Purpose**: Load pre-calculated factor scores from the factor analysis step.

### Step 2: Standardize Data for Clustering

```python
scaler = StandardScaler()
factor_data_scaled = scaler.fit_transform(factor_data)
```

**Why standardize?**
- K-means is distance-based and sensitive to scale
- Ensures each factor contributes equally to clustering
- Prevents factors with larger variance from dominating

---

## Determining Optimal Number of Clusters

### Method 1: Silhouette Score (Primary)

```python
silhouette = silhouette_score(factor_data_scaled, cluster_labels)
```

**Interpretation**:
- Range: -1 to 1
- **> 0.5**: Well-separated clusters
- **0.3-0.5**: Reasonable separation
- **< 0.3**: Weak cluster structure

### Method 2: Davies-Bouldin Index

```python
davies_bouldin = davies_bouldin_score(factor_data_scaled, cluster_labels)
```

**Interpretation**:
- Range: 0 to ∞
- **Lower is better**
- Measures average similarity between each cluster and its most similar cluster

### Method 3: Calinski-Harabasz Index

```python
calinski_harabasz = calinski_harabasz_score(factor_data_scaled, cluster_labels)
```

**Interpretation**:
- Ratio of between-cluster to within-cluster dispersion
- **Higher is better**
- Detects well-separated, compact clusters

### Method 4: Elbow Method

```python
inertia = kmeans.inertia_  # Sum of squared distances to nearest centroid
```

**Interpretation**:
- Plot inertia vs k
- Look for "elbow" point where inertia decrease slows
- Used as confirmation of other metrics

---

## K-Means Clustering

### Basic Implementation

```python
kmeans = KMeans(n_clusters=optimal_k, 
                random_state=42, 
                n_init=20)  # Try 20 different initializations
cluster_labels = kmeans.fit_predict(factor_data_scaled)
```

**Parameters**:
- `n_clusters`: Number of clusters to form
- `random_state`: Ensures reproducibility
- `n_init`: Number of times to run k-means with different initializations
- `max_iter`: Maximum number of iterations (default 300)

### Accessing Results

```python
# Cluster assignments
cluster_labels  # Array of cluster IDs for each player

# Cluster centroids
kmeans.cluster_centers_  # Position of each cluster center

# Within-cluster sum of squares
kmeans.inertia_  # Sum of squared distances
```

---

## Cluster Analysis and Characterization

### Calculate Cluster Profiles

```python
for cluster_id in range(optimal_k):
    cluster_mask = cluster_labels == cluster_id
    cluster_members = players[cluster_mask]
    cluster_factor_scores = factor_data[cluster_mask]
    
    # Calculate mean and std for each factor
    mean_scores = np.mean(cluster_factor_scores, axis=0)
    std_scores = np.std(cluster_factor_scores, axis=0)
```

### Identify Strengths and Weaknesses

```python
# Compare cluster mean to overall mean
overall_mean = np.mean(factor_data[:, i])
cluster_mean = np.mean(cluster_factor_scores[:, i])
difference = cluster_mean - overall_mean

# Classify as strength or weakness
if difference > 0.5:
    factor_strengths.append((factor, difference))
elif difference < -0.5:
    factor_weaknesses.append((factor, difference))
```

**Threshold Logic**:
- **Strength**: cluster_mean - overall_mean > 0.5 (std units above average)
- **Weakness**: cluster_mean - overall_mean < -0.5 (std units below average)
- **Average**: Between -0.5 and 0.5

---

## Automatic Cluster Naming

### Naming Strategy

The code uses hierarchical rules based on top strengths:

```python
def generate_cluster_name(strengths, weaknesses, factor_map):
    if 'Other Tournament' in top_strength_name:
        return "Tournament Specialists"
    elif 'Club Match' in top_strength_name and 'Playing Time' in top_strength_name:
        return "Club Workhorses"
    elif 'National Team' in top_strength_name and 'Selection' in top_strength_name:
        return "National Team Leaders"
    elif 'Attacking' in top_strength_name or 'Scoring' in top_strength_name:
        return "Attacking Specialists"
    elif 'Physical' in top_strength_name:
        return "Physically Dominant"
    return "Balanced Performers"
```

### Cluster Names Used

| Strength | Cluster Name |
|----------|--------------|
| Other Tournament Performance | Tournament Specialists |
| Club Match Frequency & Playing Time | Club Workhorses |
| National Team Performance & Selection | National Team Leaders |
| Scoring/Attacking Contribution | Attacking Specialists |
| Physical Attributes | Physically Dominant |
| Multiple/None | Balanced Performers |

---

## Three-Sentence Description Generation

### Overview Sentence

```python
def generate_overview_sentence(cluster_id, profile, factor_map):
    size = profile['size']
    size_desc = "large" if size > 25 else "medium" if size > 15 else "small"
    
    top_factor = factor_map.get(strengths[0][0], strengths[0][0])
    return f"This {size_desc} cluster of {size} players is characterized by excellence in {top_factor.lower()}."
```

**Pattern**: `"This [size] cluster of [n] players is characterized by excellence in [top factor]."`

### Strengths Sentence

```python
def generate_strengths_sentence(strengths, factor_map):
    if len(strengths) >= 2:
        factor1 = factor_map.get(strengths[0][0], strengths[0][0]).lower()
        factor2 = factor_map.get(strengths[1][0], strengths[1][0]).lower()
        return f"Key strengths include {factor1} and {factor2}."
    else:
        factor = factor_map.get(strengths[0][0], strengths[0][0]).lower()
        return f"The primary strength of this cluster is {factor}."
```

**Patterns**:
- With 2+ strengths: `"Key strengths include [factor1] and [factor2]."`
- With 1 strength: `"The primary strength is [factor]."`
- Without strengths: `"The cluster shows no particularly distinctive strengths."`

### Weaknesses Sentence

```python
def generate_weaknesses_sentence(weaknesses, factor_map):
    if len(weaknesses) >= 2:
        factor1 = factor_map.get(weaknesses[0][0], weaknesses[0][0]).lower()
        factor2 = factor_map.get(weaknesses[1][0], weaknesses[1][0]).lower()
        return f"Notable limitations include {factor1} and {factor2}."
    else:
        factor = factor_map.get(weaknesses[0][0], weaknesses[0][0]).lower()
        return f"A notable limitation is {factor}."
```

**Patterns**:
- With 2+ weaknesses: `"Notable limitations include [factor1] and [factor2]."`
- With 1 weakness: `"A notable limitation is [factor]."`
- Without weaknesses: `"The cluster demonstrates no significant weaknesses."`

---

## Visualizations

### 1. Cluster Evaluation Metrics (3 subplots)

```python
# Elbow Curve
ax.plot(K_range, inertias, 'bo-')

# Silhouette Score
ax.plot(K_range, silhouette_scores, 'go-')

# Davies-Bouldin Index
ax.plot(K_range, davies_bouldin_scores, 'mo-')
```

### 2. Cluster Size Distribution

```python
cluster_sizes = [np.sum(cluster_labels == i) for i in range(optimal_k)]
ax.bar(range(optimal_k), cluster_sizes)
```

### 3. 2D Scatter Plots

```python
# Factor 1 vs Factor 2
for i in range(optimal_k):
    mask = cluster_labels == i
    ax.scatter(factor_data_scaled[mask, 0], factor_data_scaled[mask, 1])

# Plot centroids
ax.scatter(kmeans_final.cluster_centers_[:, 0], 
          kmeans_final.cluster_centers_[:, 1], 
          marker='X', s=400, c='red')
```

### 4. Cluster Profile Heatmap

```python
cluster_means = np.array([cluster_profiles[i]['mean_scores'] 
                          for i in range(optimal_k)])
im = ax.imshow(cluster_means, cmap='RdBu_r', vmin=-2, vmax=2)

# Add text annotations
for i in range(optimal_k):
    for j in range(len(factor_columns)):
        ax.text(j, i, f'{cluster_means[i, j]:.1f}',
               ha="center", va="center", fontweight='bold')
```

### 5. Box Plots

```python
box_data = [factor_data_scaled[cluster_labels == i, 0] 
            for i in range(optimal_k)]
bp = ax.boxplot(box_data, labels=[f'C{i}' for i in range(optimal_k)])
```

---

## Output Files and Data Structures

### 1. cluster_assignments.csv

```python
cluster_assignment_df = pd.DataFrame({
    'Player': players,
    'Cluster': cluster_labels,
    'Cluster Name': [cluster_names[c] for c in cluster_labels]
})
```

**Columns**:
- `Player`: Player name
- `Cluster`: Cluster ID (0 to k-1)
- `Cluster Name`: Descriptive cluster name

### 2. player_factor_cluster_results.csv

```python
full_results = factor_scores_df.copy()
full_results['Cluster'] = cluster_labels
full_results['Cluster Name'] = [cluster_names[c] for c in cluster_labels]
```

**Contains**: All factor scores + cluster assignments

### 3. cluster_descriptions.txt

Text file with detailed descriptions:
- Cluster name
- 3-sentence description
- All cluster members listed
- Factor strengths and weaknesses
- Percentage of total dataset

### 4. cluster_statistics.txt

Statistical summary:
- Silhouette, Davies-Bouldin, Calinski-Harabasz scores
- Quality metrics for all k values tested
- Cluster profiles with mean factor scores

---

## Advanced Usage

### Extract Specific Cluster Members

```python
cluster_0_members = players[cluster_labels == 0]
cluster_0_scores = factor_data_scaled[cluster_labels == 0]
```

### Calculate Inter-Cluster Distance

```python
from scipy.spatial.distance import pdist, squareform

# Distance between cluster centroids
centroid_distances = pdist(kmeans.cluster_centers_, metric='euclidean')
distance_matrix = squareform(centroid_distances)
```

### Custom Cluster Analysis

```python
# Get most representative player (closest to centroid)
distances = np.linalg.norm(cluster_factor_scores - centroid, axis=1)
closest_idx = np.argsort(distances)[0]
representative_player = cluster_members[closest_idx]

# Get outlier player (farthest from centroid)
outlier_idx = np.argsort(distances)[-1]
outlier_player = cluster_members[outlier_idx]
```

### Hierarchical Clustering for Comparison

```python
from scipy.cluster.hierarchy import dendrogram, linkage

# Compute linkage matrix
linkage_matrix = linkage(factor_data_scaled, method='ward')

# Plot dendrogram
dendrogram(linkage_matrix)
```

---

## Common Issues and Solutions

### Issue 1: Poor Clustering (Low Silhouette Score)

**Causes**:
- Optimal k doesn't match data structure
- Factors have limited discriminative power
- Data is highly overlapped

**Solutions**:
```python
# Try wider k range
K_range = range(2, 20)

# Use different initialization
kmeans = KMeans(n_clusters=k, n_init=50, random_state=42)

# Try different distance metrics (use custom implementation)
```

### Issue 2: Cluster Sizes are Unbalanced

**Cause**: K-means tends to create similarly-sized clusters

**Solution**: Use clustering methods that allow variable cluster sizes:
```python
from sklearn.cluster import DBSCAN
# DBSCAN doesn't require specifying number of clusters
```

### Issue 3: Reproducibility Issues

**Solution**: Always set random_state:
```python
kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
np.random.seed(42)
```

---

## Mathematical Background

### K-Means Algorithm

```
1. Initialize k cluster centers randomly
2. Repeat until convergence:
   a. Assign each point to nearest centroid
   b. Update centroids as mean of assigned points
   
Minimize: Σ_i Σ_j ||x_j - μ_i||²
```

### Silhouette Coefficient

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))

Where:
a(i) = average distance to points in same cluster
b(i) = minimum average distance to points in other clusters
```

### Davies-Bouldin Index

```
DB = (1/k) Σ_i max_{j≠i} (σ_i + σ_j) / d(c_i, c_j)

Where:
σ_i = average distance of points in cluster i to centroid
d(c_i, c_j) = distance between centroids
```

---

## Customization Guide

### Change Number of Clusters Manually

```python
# Override automatic selection
final_k = 5  # Instead of optimal_k
kmeans_final = KMeans(n_clusters=final_k, random_state=42, n_init=20)
```

### Modify Cluster Names

```python
# Replace the naming function
def custom_name(strengths, weaknesses, factor_map):
    return "Your Custom Name"

cluster_names[cluster_id] = custom_name(...)
```

### Add Custom Metrics

```python
from sklearn.metrics import dunn_index, silhouette_samples

# Per-sample silhouette scores
silhouette_vals = silhouette_samples(factor_data_scaled, cluster_labels)
```

### Change Visualization Colors

```python
# Use different colormap
colors = plt.cm.viridis(np.linspace(0, 1, optimal_k))
# or
colors = plt.cm.tab10(np.arange(optimal_k))
```

---

## Key Metrics Summary

| Metric | Best Value | Interpretation |
|--------|-----------|-----------------|
| Silhouette Score | Close to 1 | Well-separated clusters |
| Davies-Bouldin Index | Low (close to 0) | Compact, separated clusters |
| Calinski-Harabasz | High | Dense, well-separated clusters |
| Inertia | Low | Tight within-cluster grouping |

---

## References

- K-means: MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations"
- Silhouette: Rousseeuw, P. J. (1987). "Silhouettes: A graphical aid to the interpretation and validation"
- Davies-Bouldin: Davies, D. L., & Bouldin, D. W. (1979). "A cluster separation measure"
- Calinski-Harabasz: Caliński, T., & Harabasz, J. (1974). "A dendrite method for cluster analysis"

---

## Performance Considerations

### Time Complexity
- K-means: O(nkdi) where n=samples, k=clusters, d=dimensions, i=iterations
- For rugby data: O(97 × 9 × 9 × 100) ≈ manageable

### Space Complexity
- O(nk) for storing cluster assignments
- O(kd) for storing centroids
- Minimal for this dataset

### Optimization Tips
```python
# Faster convergence
kmeans = KMeans(n_clusters=k, 
                algorithm='elkan',  # Better algorithm
                n_init=10,           # Fewer initializations
                max_iter=300)

# Parallel processing
kmeans = KMeans(n_clusters=k, n_jobs=-1)  # Use all CPUs
```

---

## Contact & Support

For questions about modifications or additional analyses, ensure:
1. Data is properly standardized before clustering
2. Optimal k is selected based on multiple metrics
3. Results are validated with silhouette/DB scores
4. Cluster sizes are reasonable (avoid k too large)
