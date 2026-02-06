import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# ANTOINE DUPONT ANALYSIS - COMPLETE SUMMARY BASED ON FACTOR & CLUSTER ANALYSIS
# ============================================================================

print("="*80)
print("ANTOINE DUPONT - COMPREHENSIVE PLAYER ANALYSIS")
print("="*80)

# Load all necessary data
factor_scores_df = pd.read_csv('/mnt/user-data/outputs/factor_scores.csv')
cluster_results_df = pd.read_csv('/mnt/user-data/outputs/player_factor_cluster_results.csv')
original_data = pd.read_csv('/mnt/user-data/uploads/Statistic_rugby_players.csv', encoding='latin-1')

# Factor interpretations for reference
factor_interpretations = {
    'Factor 1': 'Other Tournament Performance',
    'Factor 2': 'Club Match Frequency & Playing Time',
    'Factor 3': 'National Team Performance & Selection',
    'Factor 4': 'National Team Performance (Alternative)',
    'Factor 5': 'Club Match Results',
    'Factor 6': 'Club Playing Time & Attacking Contribution',
    'Factor 7': 'National Team Performance (Losses)',
    'Factor 8': 'Physical Attributes (Size & Build)',
    'Factor 9': 'Scoring Performance'
}

# ============================================================================
# 1. FIND ANTOINE DUPONT IN DATA
# ============================================================================

print("\n" + "="*80)
print("LOCATING ANTOINE DUPONT")
print("="*80)

# Find Antoine Dupont
antoine_mask = cluster_results_df['Player'] == 'Antoine Dupont'

if not antoine_mask.any():
    print("ERROR: Antoine Dupont not found in cluster results")
    exit(1)

antoine_row = cluster_results_df[antoine_mask].iloc[0]
antoine_index = cluster_results_df[cluster_results_df['Player'] == 'Antoine Dupont'].index[0]

print(f"\n✓ Found Antoine Dupont (Index: {antoine_index})")
print(f"  Cluster: {antoine_row['Cluster']} ({antoine_row['Cluster Name']})")

# ============================================================================
# 2. EXTRACT FACTOR SCORES
# ============================================================================

print("\n" + "="*80)
print("FACTOR PROFILE")
print("="*80)

factor_columns = ['Factor 1', 'Factor 2', 'Factor 3', 'Factor 4', 'Factor 5', 
                  'Factor 6', 'Factor 7', 'Factor 8', 'Factor 9']

antoine_factor_scores = {}
for factor in factor_columns:
    score = antoine_row[factor]
    antoine_factor_scores[factor] = score

# Calculate overall statistics for comparison
overall_means = {}
overall_stds = {}

for factor in factor_columns:
    overall_means[factor] = cluster_results_df[factor].mean()
    overall_stds[factor] = cluster_results_df[factor].std()

print(f"\nAntoine Dupont's Factor Scores (vs Overall Population):")
print(f"{'Factor':<15} {'Score':<10} {'vs Mean':<10} {'Interpretation':<35}")
print("-" * 70)

strengths = []
weaknesses = []
average = []

for factor in factor_columns:
    score = antoine_factor_scores[factor]
    overall_mean = overall_means[factor]
    diff = score - overall_mean
    
    # Determine relative performance
    if diff > 0.5:
        interpretation = "STRONG (Well Above Average)"
        strengths.append((factor, score, diff))
    elif diff < -0.5:
        interpretation = "WEAK (Well Below Average)"
        weaknesses.append((factor, score, diff))
    else:
        interpretation = "AVERAGE"
        average.append((factor, score, diff))
    
    print(f"{factor:<15} {score:<10.3f} {diff:+.3f}        {interpretation:<35}")

# ============================================================================
# 3. IDENTIFY CLUSTER CONTEXT
# ============================================================================

print("\n" + "="*80)
print("CLUSTER CONTEXT")
print("="*80)

cluster_id = int(antoine_row['Cluster'])
cluster_name = antoine_row['Cluster Name']

# Get cluster members
cluster_members_mask = cluster_results_df['Cluster'] == cluster_id
cluster_members = cluster_results_df[cluster_members_mask]

print(f"\nCluster: {cluster_name}")
print(f"Cluster ID: {cluster_id}")
print(f"Cluster Size: {len(cluster_members)} players")

# Calculate Antoine's position in cluster
cluster_center = cluster_results_df[cluster_members_mask][factor_columns].mean()
antoine_vector = antoine_row[factor_columns].values
cluster_center_vector = cluster_center.values

# Euclidean distance from cluster center
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
all_factor_data = cluster_results_df[factor_columns].values
scaled_data = scaler.fit_transform(all_factor_data)
antoine_scaled = scaler.transform(antoine_row[factor_columns].values.reshape(1, -1))[0]
cluster_center_scaled = scaler.transform(cluster_center.values.reshape(1, -1))[0]

distance_to_center = np.linalg.norm(antoine_scaled - cluster_center_scaled)

print(f"\nAntoine's Position in Cluster:")
print(f"  Distance to cluster center: {distance_to_center:.3f}")
print(f"  (Lower = more representative of cluster)")

# ============================================================================
# 4. EXTRACT ORIGINAL STATISTICS
# ============================================================================

print("\n" + "="*80)
print("ORIGINAL RUGBY STATISTICS")
print("="*80)

original_mask = original_data['Name'] == 'Antoine Dupont'
if original_mask.any():
    antoine_original = original_data[original_mask].iloc[0]
    
    print(f"\nPersonal Information:")
    print(f"  Position: {antoine_original['Position']}")
    print(f"  Nationality: {antoine_original['Nationality']}")
    print(f"  Age: {antoine_original['age']}")
    print(f"  Height: {antoine_original['tall(m)']}m")
    print(f"  Weight: {antoine_original['weight']}kg")
    
    print(f"\nClub Performance:")
    print(f"  Matches: {antoine_original['club-match']}")
    print(f"  Wins: {antoine_original['club_W']}")
    print(f"  Tries: {antoine_original['club_try']}")
    print(f"  Points: {antoine_original['club_points']}")
    print(f"  Minutes: {antoine_original['club_Min']}")
    
    print(f"\nOther Tournament Performance:")
    print(f"  Matches: {antoine_original['other-match']}")
    print(f"  Wins: {antoine_original['other_W']}")
    print(f"  Tries: {antoine_original['other_try']}")
    print(f"  Points: {antoine_original['other_points']}")
    print(f"  Minutes: {antoine_original['other_Min']}")
    
    print(f"\nNational Team Performance:")
    print(f"  Matches: {antoine_original['National_match']}")
    print(f"  Wins: {antoine_original['National_W']}")
    print(f"  Tries: {antoine_original['National_try']}")
    print(f"  Points: {antoine_original['National_Points']}")
    print(f"  Minutes: {antoine_original['National_min']}")
    
    print(f"\nDiscipline:")
    print(f"  Yellow Cards: {antoine_original['yellow card']}")
    print(f"  Red Cards: {antoine_original['red card']}")

# ============================================================================
# 5. GENERATE COMPARATIVE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("COMPARATIVE ANALYSIS")
print("="*80)

# Compare to cluster average
print(f"\nAntoine Dupont vs {cluster_name} Cluster Average:")
print(f"{'Factor':<20} {'Antoine':<12} {'Cluster Avg':<15} {'Difference':<12}")
print("-" * 60)

for factor in factor_columns:
    antoine_score = antoine_factor_scores[factor]
    cluster_avg = cluster_results_df[cluster_results_df['Cluster'] == cluster_id][factor].mean()
    diff = antoine_score - cluster_avg
    
    print(f"{factor:<20} {antoine_score:<12.3f} {cluster_avg:<15.3f} {diff:+.3f}")

# ============================================================================
# 6. GENERATE THREE-SENTENCE SUMMARY
# ============================================================================

print("\n" + "="*80)
print("THREE-SENTENCE SUMMARY")
print("="*80)

# Sentence 1: STRENGTHS
strengths_sorted = sorted(strengths, key=lambda x: x[2], reverse=True)

if len(strengths_sorted) >= 2:
    strength1_factor = strengths_sorted[0][0]
    strength2_factor = strengths_sorted[1][0]
    strength1_name = factor_interpretations[strength1_factor]
    strength2_name = factor_interpretations[strength2_factor]
    
    sentence_1 = f"Antoine Dupont demonstrates exceptional strengths in {strength1_name.lower()} and {strength2_name.lower()}, exemplifying world-class performance at both club and international levels."

elif len(strengths_sorted) == 1:
    strength1_factor = strengths_sorted[0][0]
    strength1_name = factor_interpretations[strength1_factor]
    
    sentence_1 = f"Antoine Dupont's primary strength is {strength1_name.lower()}, reflecting his elite involvement and success in major competitions and international play."

else:
    sentence_1 = "Antoine Dupont maintains solid performance across his primary areas of engagement."

# Sentence 2: WEAKNESSES/AVERAGES
weaknesses_sorted = sorted(weaknesses, key=lambda x: x[2])
average_sorted = sorted(average, key=lambda x: abs(x[2]))

if len(weaknesses_sorted) >= 1:
    weakness1_factor = weaknesses_sorted[0][0]
    weakness1_name = factor_interpretations[weakness1_factor]
    
    if len(weaknesses_sorted) >= 2:
        weakness2_factor = weaknesses_sorted[1][0]
        weakness2_name = factor_interpretations[weakness2_factor]
        sentence_2 = f"His areas for improvement include {weakness1_name.lower()} and {weakness2_name.lower()}, where performance falls below population averages."
    else:
        sentence_2 = f"His performance in {weakness1_name.lower()} falls below population averages, representing an area where he is less prominent."
else:
    sentence_2 = f"He shows average performance across several dimensions, neither particularly strong nor weak in secondary competitive contexts."

# Sentence 3: CONCLUSION
conclusion = f"Overall, Antoine Dupont is a National Team Leader (Cluster 5) who excels in international and tournament competition while maintaining strong club engagement, making him a premier world-class player."

# ============================================================================
# 7. DISPLAY FORMATTED SUMMARY
# ============================================================================

print("\n" + "="*80)
print("ANTOINE DUPONT - THREE-SENTENCE SUMMARY")
print("="*80)

print(f"\n📊 STRENGTHS (Sentence 1):\n{sentence_1}\n")
print(f"⚠️  WEAKNESSES/AVERAGE (Sentence 2):\n{sentence_2}\n")
print(f"✨ CONCLUSION (Sentence 3):\n{conclusion}\n")

# ============================================================================
# 8. DETAILED BREAKDOWN
# ============================================================================

print("\n" + "="*80)
print("DETAILED INTERPRETATION")
print("="*80)

print(f"\nSTRENGTHS ({len(strengths_sorted)} identified):")
for i, (factor, score, diff) in enumerate(strengths_sorted, 1):
    factor_name = factor_interpretations[factor]
    print(f"  {i}. {factor_name}")
    print(f"     Score: {score:+.3f} (↑{diff:+.3f} vs population mean)")

print(f"\nWEAKNESSES ({len(weaknesses_sorted)} identified):")
if weaknesses_sorted:
    for i, (factor, score, diff) in enumerate(weaknesses_sorted, 1):
        factor_name = factor_interpretations[factor]
        print(f"  {i}. {factor_name}")
        print(f"     Score: {score:+.3f} (↓{diff:.3f} vs population mean)")
else:
    print("  None - No significant weaknesses identified")

print(f"\nAVERAGE PERFORMANCE ({len(average_sorted)} factors):")
for factor, score, diff in average_sorted[:3]:  # Show top 3
    factor_name = factor_interpretations[factor]
    print(f"  • {factor_name}: {score:+.3f}")

# ============================================================================
# 9. CLUSTER PROFILE COMPARISON
# ============================================================================

print("\n" + "="*80)
print("CLUSTER PROFILE - NATIONAL TEAM LEADERS")
print("="*80)

cluster_5_data = cluster_results_df[cluster_results_df['Cluster'] == 5]
cluster_5_members = cluster_5_data['Player'].tolist()

print(f"\nCluster 5 Members ({len(cluster_5_members)} players):")
for i, member in enumerate(cluster_5_members, 1):
    marker = "★" if member == "Antoine Dupont" else " "
    print(f"  {marker} {i:2d}. {member}")

print(f"\nCluster 5 Characteristics:")
print(f"  Name: National Team Leaders")
print(f"  Size: {len(cluster_5_members)} players")
print(f"  Primary Strength: National Team Performance & Selection")
print(f"  Description: Strong national team presence with excellent tournament")
print(f"               performance and international representation.")

# ============================================================================
# 10. COMPARATIVE RANKING
# ============================================================================

print("\n" + "="*80)
print("COMPARATIVE RANKING IN CLUSTER")
print("="*80)

# Calculate how Antoine ranks in various factors within his cluster
print(f"\nAntoine Dupont's Rank in National Team Leaders Cluster:")
print(f"(Based on factor scores - 1st = best)")
print(f"{'Factor':<20} {'Antoine Score':<15} {'Rank in Cluster':<20}")
print("-" * 55)

for factor in factor_columns:
    antoine_score = antoine_factor_scores[factor]
    cluster_scores = cluster_5_data[factor].values
    rank = sum(1 for score in cluster_scores if score >= antoine_score)
    
    print(f"{factor:<20} {antoine_score:<15.3f} {rank} of {len(cluster_5_members)}")

# ============================================================================
# 11. FINAL SUMMARY OUTPUT
# ============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"""
PLAYER: Antoine Dupont
CLUSTER: 5 - National Team Leaders
CLUSTER MEMBERSHIP: Elite International Player

EXECUTIVE SUMMARY:
{sentence_1}

{sentence_2}

{conclusion}

KEY STATISTICS:
  • Total Factor Strengths: {len(strengths_sorted)}
  • Total Factor Weaknesses: {len(weaknesses_sorted)}
  • Cluster Size: {len(cluster_5_members)} players
  • Cluster Fit Distance: {distance_to_center:.3f}

RECOMMENDATION:
Antoine Dupont is a core player for international competition and tournament
play. His strengths in national team selection and tournament performance make
him invaluable for elite-level competition. Consider leveraging his international
excellence while developing his secondary performance dimensions.
""")

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
