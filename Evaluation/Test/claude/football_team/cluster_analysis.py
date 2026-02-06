import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

# ==================== Load Data ====================
print("="*70)
print("LOADING DATA")
print("="*70)

# Load factor scores
factor_scores = pd.read_csv('/mnt/user-data/outputs/factor_scores.csv')
print(f"Factor scores shape: {factor_scores.shape}")
print(f"Factors: {list(factor_scores.columns)}")

# Load original data for team names
df_original = pd.read_csv('/mnt/user-data/uploads/team_stats.csv')
team_names = df_original['club_name'].values

# Load metric map for descriptions
with open('/mnt/user-data/uploads/match_api_metric_map.json', 'r') as f:
    metric_map = json.load(f)

print(f"Number of teams: {len(team_names)}")

# ==================== Determine Optimal Number of Clusters ====================
print("\n" + "="*70)
print("DETERMINING OPTIMAL NUMBER OF CLUSTERS")
print("="*70)

# Test different numbers of clusters
silhouette_scores = []
davies_bouldin_scores = []
calinski_harabasz_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(factor_scores)
    
    sil_score = silhouette_score(factor_scores, labels)
    db_score = davies_bouldin_score(factor_scores, labels)
    ch_score = calinski_harabasz_score(factor_scores, labels)
    
    silhouette_scores.append(sil_score)
    davies_bouldin_scores.append(db_score)
    calinski_harabasz_scores.append(ch_score)
    
    print(f"k={k}: Silhouette={sil_score:.4f}, Davies-Bouldin={db_score:.4f}, Calinski-Harabasz={ch_score:.4f}")

# Plot clustering metrics
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(k_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters', fontsize=11)
axes[0].set_ylabel('Silhouette Score', fontsize=11)
axes[0].set_title('Silhouette Score (Higher is Better)', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=k_range[np.argmax(silhouette_scores)], color='r', linestyle='--', alpha=0.5)

axes[1].plot(k_range, davies_bouldin_scores, 'go-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters', fontsize=11)
axes[1].set_ylabel('Davies-Bouldin Index', fontsize=11)
axes[1].set_title('Davies-Bouldin Index (Lower is Better)', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].axvline(x=k_range[np.argmin(davies_bouldin_scores)], color='r', linestyle='--', alpha=0.5)

axes[2].plot(k_range, calinski_harabasz_scores, 'mo-', linewidth=2, markersize=8)
axes[2].set_xlabel('Number of Clusters', fontsize=11)
axes[2].set_ylabel('Calinski-Harabasz Score', fontsize=11)
axes[2].set_title('Calinski-Harabasz Score (Higher is Better)', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3)
axes[2].axvline(x=k_range[np.argmax(calinski_harabasz_scores)], color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/clustering_metrics.png', dpi=300, bbox_inches='tight')
print("\n✓ Clustering metrics plot saved!")

# Determine optimal k
optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
optimal_k_davies_bouldin = k_range[np.argmin(davies_bouldin_scores)]
optimal_k_calinski = k_range[np.argmax(calinski_harabasz_scores)]

print(f"\nOptimal k (Silhouette): {optimal_k_silhouette}")
print(f"Optimal k (Davies-Bouldin): {optimal_k_davies_bouldin}")
print(f"Optimal k (Calinski-Harabasz): {optimal_k_calinski}")

# Use consensus - most common optimal k
optimal_k = optimal_k_silhouette  # Silhouette is most interpretable
print(f"\n>>> Selected optimal number of clusters: {optimal_k} <<<")

# ==================== Perform Final Clustering ====================
print("\n" + "="*70)
print(f"CLUSTERING WITH {optimal_k} CLUSTERS")
print("="*70)

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(factor_scores)

# Add cluster labels to dataframe
factor_scores['Cluster'] = cluster_labels
factor_scores['Team'] = team_names

print(f"\nCluster distribution:")
cluster_counts = factor_scores['Cluster'].value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    percentage = (count / len(factor_scores)) * 100
    print(f"  Cluster {cluster_id}: {count} teams ({percentage:.1f}%)")

# ==================== Analyze Cluster Characteristics ====================
print("\n" + "="*70)
print("CLUSTER CHARACTERISTICS ANALYSIS")
print("="*70)

# Calculate cluster centers and statistics
cluster_profiles = {}
cluster_descriptions = {}

for cluster_id in range(optimal_k):
    cluster_data = factor_scores[factor_scores['Cluster'] == cluster_id]
    teams_in_cluster = cluster_data['Team'].values
    
    print(f"\n{'='*70}")
    print(f"CLUSTER {cluster_id}")
    print(f"{'='*70}")
    print(f"Teams ({len(teams_in_cluster)}): {', '.join(teams_in_cluster)}")
    
    # Get mean scores for each factor
    factor_cols = [col for col in factor_scores.columns if col not in ['Cluster', 'Team']]
    cluster_mean_scores = cluster_data[factor_cols].mean()
    
    print(f"\nFactor Scores (Mean):")
    cluster_mean_scores_sorted = cluster_mean_scores.sort_values(ascending=False)
    for factor, score in cluster_mean_scores_sorted.items():
        print(f"  {factor}: {score:.4f}")
    
    # Identify top strengths (highest positive factors)
    top_strengths = cluster_mean_scores_sorted.head(3)
    top_weaknesses = cluster_mean_scores_sorted.tail(3)
    
    print(f"\nTop Strengths:")
    for factor, score in top_strengths.items():
        print(f"  + {factor}: {score:.4f}")
    
    print(f"\nTop Weaknesses:")
    for factor, score in top_weaknesses.items():
        print(f"  - {factor}: {score:.4f}")
    
    cluster_profiles[cluster_id] = {
        'size': len(teams_in_cluster),
        'teams': list(teams_in_cluster),
        'mean_scores': cluster_mean_scores.to_dict(),
        'top_strengths': top_strengths.index.tolist(),
        'top_weaknesses': top_weaknesses.index.tolist(),
        'strength_values': top_strengths.values.tolist(),
        'weakness_values': top_weaknesses.values.tolist()
    }

# ==================== Name and Describe Clusters ====================
print("\n" + "="*70)
print("NAMING AND DESCRIBING CLUSTERS")
print("="*70)

cluster_names = {}
cluster_full_descriptions = {}

def get_cluster_name_and_description(cluster_id, strengths, weaknesses, cluster_profiles):
    """Generate intelligent cluster name and description based on characteristics"""
    
    # Count cluster characteristics
    attacking_strengths = sum(1 for s in strengths if 'Attacking' in s or 'Output' in s)
    defensive_strengths = sum(1 for s in strengths if 'Defensive' in s or 'Prowess' in s)
    possession_strengths = sum(1 for s in strengths if 'Possession' in s or 'Control' in s)
    recovery_strengths = sum(1 for s in strengths if 'Recovery' in s or 'Transition' in s)
    box_strengths = sum(1 for s in strengths if 'Box' in s)
    
    attacking_weaknesses = sum(1 for w in weaknesses if 'Attacking' in w or 'Output' in w)
    defensive_weaknesses = sum(1 for w in weaknesses if 'Defensive' in w or 'Prowess' in w)
    possession_weaknesses = sum(1 for w in weaknesses if 'Possession' in w or 'Control' in w)
    
    # Determine primary characteristics
    profile = cluster_profiles[cluster_id]
    strength_vals = profile['strength_values']
    weakness_vals = profile['weakness_values']
    
    # Elite teams (multiple attacking strengths, positive defensive)
    if attacking_strengths >= 2 and defensive_strengths >= 1 and strength_vals[0] > 1.5:
        name = "Elite Attackers"
        overview = "This cluster represents teams that excel in offensive prowess, demonstrating exceptional attacking capability and ability to create high-quality scoring opportunities consistently."
        strengths_desc = "These teams are characterized by superior goal-scoring efficiency, high shot volume, excellent expected points generation, strong threat creation, and swift recovery enabling rapid transitions to attacks."
        weaknesses_desc = "However, these teams may show occasional defensive vulnerabilities when facing organized opposition, potentially conceding from transitions and set-piece situations."
    
    # Defensive specialists
    elif defensive_strengths >= 2 and attacking_weaknesses >= 2 and strength_vals[0] > 0.8:
        name = "Defensive Stalwarts"
        overview = "This cluster represents teams that prioritize defensive excellence and structured play, establishing solid foundations built on disciplined defensive organization."
        strengths_desc = "These teams showcase outstanding defensive intensity, high pressing efficiency (PPDA), strong recovery positioning, and coordinated defensive structures that neutralize offensive threats effectively."
        weaknesses_desc = "Conversely, these teams struggle to generate attacking output, showing lower shot volume and reduced creativity in penetrating opposition defenses."
    
    # Possession-dominant teams
    elif possession_strengths >= 2 and attacking_weaknesses <= 1:
        name = "Possession Architects"
        overview = "This cluster represents teams that dictate play through superior ball control and methodical build-up, establishing dominance through possession retention and structured passing patterns."
        strengths_desc = "These teams excel at maintaining high possession percentages, controlling field tilt significantly in their favor, executing technical build-up play, and creating structured attacking movements from deep positions."
        weaknesses_desc = "Yet, these teams sometimes struggle with directness and conversion efficiency, occasionally showing vulnerability to compact defensive shapes and counter-attacking threats."
    
    # Transition/Counter specialists
    elif recovery_strengths >= 2 and possession_weaknesses >= 1:
        name = "Transition Specialists"
        overview = "This cluster represents teams that thrive on quick transitions and counter-attacking football, capitalizing on rapid possession recovery to create attacking opportunities."
        strengths_desc = "These teams demonstrate exceptional speed in regaining possession, executing lethal fast transitions within 10 seconds, distributing with directness, and penetrating final third spaces quickly following turnovers."
        weaknesses_desc = "Conversely, these teams have lower possession dominance, relying on quality counter-attacking play over sustained build-up possession, potentially exposed against methodical possession-based opponents."
    
    # Box dominators
    elif box_strengths >= 1 and strength_vals[0] > 0.7:
        name = "Box Dominators"
        overview = "This cluster represents teams that establish commanding control within the final third, generating numerous box touches and penetrating opportunities through varied attacking approaches."
        strengths_desc = "These teams excel at creating frequent box entries, generating high-volume shooting situations, establishing presence through crosses and carries, and creating dangerous opportunities from set pieces and movement play."
        weaknesses_desc = "These teams often show defensive instability outside the box, potentially vulnerable to quick counter-attacks and organized defensive units that limit space in wide areas."
    
    # Balanced teams
    elif abs(strength_vals[0] - strength_vals[1]) < 0.3 and strength_vals[0] < 0.6:
        name = "Balanced Operators"
        overview = "This cluster represents teams with well-rounded performance across tactical dimensions, showing consistency without dominant specialization in any particular area."
        strengths_desc = "These teams demonstrate reliable all-around performance, maintaining competitive levels across attacking output, defensive solidity, and possession management without critical gaps."
        weaknesses_desc = "However, these teams may lack the specialized expertise or tactical dominance found in more focused tactical groups, potentially exposed when facing highly specialized opponents."
    
    # Mixed profile teams with efficiency focus
    else:
        name = "Tactical Specialists"
        overview = "This cluster represents teams with distinctive tactical characteristics, excelling in specific domains while showing variable performance in complementary areas."
        strengths_desc = "These teams demonstrate strong proficiency in their specialized tactical areas, leveraging unique approaches to neutralize opposition and create competitive advantages within their strength domains."
        weaknesses_desc = "These teams show targeted weaknesses in complementary tactical areas, potentially vulnerable when opponents exploit less developed dimensions of their play."
    
    description = f"{overview} {strengths_desc} {weaknesses_desc}"
    return name, description

for cluster_id in range(optimal_k):
    profile = cluster_profiles[cluster_id]
    strengths = profile['top_strengths']
    weaknesses = profile['top_weaknesses']
    
    # Determine cluster characteristics
    strengths_text = ' & '.join([s.replace('_', ' ').title() for s in strengths[:2]])
    weaknesses_text = ' & '.join([w.replace('_', ' ').title() for w in weaknesses[:2]])
    
    # Generate cluster name based on characteristics
    print(f"\nCluster {cluster_id}:")
    print(f"  Strengths: {strengths_text}")
    print(f"  Weaknesses: {weaknesses_text}")
    
    # Get intelligent name and description
    name, description = get_cluster_name_and_description(cluster_id, strengths, weaknesses, cluster_profiles)
    
    cluster_names[cluster_id] = name
    cluster_full_descriptions[cluster_id] = description
    
    # Parse three sentences
    sentences = description.split('. ')
    print(f"\n  Cluster Name: {name}")
    print(f"\n  Description:")
    for i, sentence in enumerate(sentences[:3], 1):
        print(f"  {i}. {sentence.strip()}.")

# ==================== Visualization ====================
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# 1. PCA visualization of clusters (using first 2 principal components)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_scores = pca.fit_transform(factor_scores[[col for col in factor_scores.columns if col not in ['Cluster', 'Team']]])

plt.figure(figsize=(12, 8))
colors = plt.cm.Set3(np.linspace(0, 1, optimal_k))
for cluster_id in range(optimal_k):
    mask = factor_scores['Cluster'] == cluster_id
    plt.scatter(pca_scores[mask, 0], pca_scores[mask, 1], 
                c=[colors[cluster_id]], s=150, alpha=0.7, 
                edgecolors='black', linewidth=1.5,
                label=f'{cluster_names[cluster_id]} (n={sum(mask)})')

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
plt.title('Team Clusters (PCA Visualization)', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/clusters_pca_visualization.png', dpi=300, bbox_inches='tight')
print("✓ PCA cluster visualization saved!")

# 2. Cluster profiles heatmap
cluster_means = []
for cluster_id in range(optimal_k):
    cluster_data = factor_scores[factor_scores['Cluster'] == cluster_id]
    factor_cols = [col for col in factor_scores.columns if col not in ['Cluster', 'Team']]
    means = cluster_data[factor_cols].mean()
    cluster_means.append(means)

cluster_profile_df = pd.DataFrame(cluster_means, 
                                  index=[f'Cluster {i}: {cluster_names[i]}' for i in range(optimal_k)])

plt.figure(figsize=(14, 6))
sns.heatmap(cluster_profile_df, cmap='RdBu_r', center=0, annot=True, fmt='.2f', 
            cbar_kws={'label': 'Mean Factor Score'}, linewidths=0.5)
plt.title('Cluster Profiles - Mean Factor Scores', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Factors', fontsize=12)
plt.ylabel('Clusters', fontsize=12)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/cluster_profiles_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Cluster profiles heatmap saved!")

# 3. Radar chart for each cluster
from math import pi

num_factors = len([col for col in factor_scores.columns if col not in ['Cluster', 'Team']])
angles = [n / float(num_factors) * 2 * pi for n in range(num_factors)]
angles += angles[:1]

factor_cols = [col for col in factor_scores.columns if col not in ['Cluster', 'Team']]

# Calculate grid size for subplots
grid_size = int(np.ceil(np.sqrt(optimal_k)))
fig, axes = plt.subplots(grid_size, grid_size, figsize=(4*grid_size, 4*grid_size), 
                         subplot_kw=dict(projection='polar'))
axes = axes.flatten() if optimal_k > 1 else [axes]

for cluster_id in range(optimal_k):
    cluster_data = factor_scores[factor_scores['Cluster'] == cluster_id]
    values = cluster_data[factor_cols].mean().values.tolist()
    values += values[:1]
    
    ax = axes[cluster_id]
    ax.plot(angles, values, 'o-', linewidth=2, color=colors[cluster_id], markersize=6)
    ax.fill(angles, values, alpha=0.25, color=colors[cluster_id])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.replace('_', '\n')[:12] for f in factor_cols], fontsize=7)
    ax.set_ylim(-2, 2)
    ax.set_title(f'Cluster {cluster_id}: {cluster_names[cluster_id]}', 
                 fontsize=10, fontweight='bold', pad=20)
    ax.grid(True)

# Hide unused subplots
for idx in range(optimal_k, len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/cluster_profiles_radar.png', dpi=300, bbox_inches='tight')
print("✓ Radar chart visualization saved!")

# 4. Cluster size distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cluster_sizes = factor_scores['Cluster'].value_counts().sort_index()
axes[0].bar(range(optimal_k), cluster_sizes.values, color=colors, edgecolor='black', linewidth=1.5)
axes[0].set_xlabel('Cluster', fontsize=12)
axes[0].set_ylabel('Number of Teams', fontsize=12)
axes[0].set_title('Cluster Size Distribution', fontsize=13, fontweight='bold')
axes[0].set_xticks(range(optimal_k))
axes[0].set_xticklabels([f'{cluster_names[i]}' for i in range(optimal_k)], rotation=45, ha='right')
axes[0].grid(True, alpha=0.3, axis='y')

# Pie chart
axes[1].pie(cluster_sizes.values, labels=[f'{cluster_names[i]}\n(n={cluster_sizes[i]})' for i in range(optimal_k)],
            colors=colors, autopct='%1.1f%%', startangle=90)
axes[1].set_title('Cluster Distribution', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/cluster_sizes.png', dpi=300, bbox_inches='tight')
print("✓ Cluster size distribution saved!")

# ==================== Save Detailed Results ====================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Save cluster assignments
cluster_assignment = pd.DataFrame({
    'Team': factor_scores['Team'],
    'Cluster_ID': factor_scores['Cluster'],
    'Cluster_Name': [cluster_names[c] for c in factor_scores['Cluster']]
})
cluster_assignment.to_csv('/mnt/user-data/outputs/team_cluster_assignments.csv', index=False)
print("✓ Team cluster assignments saved!")

# Save cluster profiles
cluster_profile_df.to_csv('/mnt/user-data/outputs/cluster_profiles.csv')
print("✓ Cluster profiles saved!")

# Save cluster descriptions and names
cluster_summary = {
    str(cluster_id): {
        'name': cluster_names[cluster_id],
        'size': cluster_profiles[cluster_id]['size'],
        'teams': cluster_profiles[cluster_id]['teams'],
        'description': cluster_full_descriptions[cluster_id],
        'mean_scores': cluster_profiles[cluster_id]['mean_scores'],
        'top_strengths': cluster_profiles[cluster_id]['top_strengths'],
        'top_weaknesses': cluster_profiles[cluster_id]['top_weaknesses']
    }
    for cluster_id in range(optimal_k)
}

with open('/mnt/user-data/outputs/cluster_summary.json', 'w') as f:
    json.dump(cluster_summary, f, indent=2)
print("✓ Cluster summary saved to JSON!")

# Save comprehensive report
with open('/mnt/user-data/outputs/cluster_analysis_report.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("CLUSTER ANALYSIS REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write("OPTIMAL CLUSTERING CONFIGURATION\n")
    f.write("-"*80 + "\n")
    f.write(f"Number of Clusters: {optimal_k}\n")
    f.write(f"Silhouette Score: {silhouette_scores[optimal_k-2]:.4f}\n")
    f.write(f"Davies-Bouldin Index: {davies_bouldin_scores[optimal_k-2]:.4f}\n")
    f.write(f"Calinski-Harabasz Score: {calinski_harabasz_scores[optimal_k-2]:.4f}\n\n")
    
    for cluster_id in range(optimal_k):
        f.write("="*80 + "\n")
        f.write(f"CLUSTER {cluster_id}: {cluster_names[cluster_id].upper()}\n")
        f.write("="*80 + "\n\n")
        
        profile = cluster_profiles[cluster_id]
        f.write(f"Cluster Size: {profile['size']} teams ({(profile['size']/len(factor_scores))*100:.1f}%)\n\n")
        
        f.write(f"Teams in Cluster:\n")
        for i, team in enumerate(profile['teams'], 1):
            f.write(f"  {i}. {team}\n")
        f.write("\n")
        
        f.write("CLUSTER DESCRIPTION\n")
        f.write("-"*80 + "\n")
        sentences = cluster_full_descriptions[cluster_id].split('. ')
        for i, sentence in enumerate(sentences, 1):
            if sentence.strip():
                f.write(f"{i}. {sentence.strip()}.\n")
        f.write("\n")
        
        f.write("TOP STRENGTHS\n")
        f.write("-"*80 + "\n")
        for i, (strength, value) in enumerate(zip(profile['top_strengths'], profile['strength_values']), 1):
            f.write(f"  {i}. {strength}: {value:.4f}\n")
        f.write("\n")
        
        f.write("TOP WEAKNESSES\n")
        f.write("-"*80 + "\n")
        for i, (weakness, value) in enumerate(zip(profile['top_weaknesses'], profile['weakness_values']), 1):
            f.write(f"  {i}. {weakness}: {value:.4f}\n")
        f.write("\n")

print("✓ Comprehensive report saved!")

# Save factor scores with cluster assignments
factor_scores_with_clusters = factor_scores.copy()
factor_scores_with_clusters['Cluster_Name'] = factor_scores_with_clusters['Cluster'].map(cluster_names)
factor_scores_with_clusters.to_csv('/mnt/user-data/outputs/factor_scores_with_clusters.csv', index=False)
print("✓ Factor scores with cluster assignments saved!")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print("\nOutput files created:")
print("  1. clustering_metrics.png - Metrics for optimal k determination")
print("  2. clusters_pca_visualization.png - PCA visualization of clusters")
print("  3. cluster_profiles_heatmap.png - Heatmap of cluster profiles")
print("  4. cluster_profiles_radar.png - Radar charts for each cluster")
print("  5. cluster_sizes.png - Cluster size distribution")
print("  6. team_cluster_assignments.csv - Team to cluster mapping")
print("  7. cluster_profiles.csv - Mean factor scores for each cluster")
print("  8. cluster_summary.json - Comprehensive cluster information")
print("  9. cluster_analysis_report.txt - Detailed text report")
print(" 10. factor_scores_with_clusters.csv - Factor scores with cluster labels")
print("\n" + "="*70)
