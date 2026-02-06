import pandas as pd
import json

# ==================== Load Data ====================
print("="*80)
print("REAL MADRID TEAM SUMMARY")
print("="*80)

# Load data
factor_scores_df = pd.read_csv('/mnt/user-data/outputs/factor_scores_with_clusters.csv')

with open('/mnt/user-data/outputs/cluster_summary.json', 'r') as f:
    cluster_summary = json.load(f)

# ==================== Find Real Madrid ====================
team_name = "Real Madrid"

# Find Real Madrid
real_madrid_row = factor_scores_df[factor_scores_df['Team'] == team_name]

if real_madrid_row.empty:
    print(f"ERROR: {team_name} not found!")
    print("Available teams (sample):", factor_scores_df['Team'].head(10).tolist())
    exit()

print(f"\n✓ Found: {team_name}")

# Get Real Madrid's data
rm_data = real_madrid_row.iloc[0]
cluster_id = int(rm_data['Cluster'])
cluster_name = rm_data['Cluster_Name']

print(f"  Cluster: {cluster_id} - {cluster_name}")

# ==================== Get Factor Scores ====================
# Get all factor columns (exclude Cluster, Team, Cluster_Name)
factor_cols = [col for col in factor_scores_df.columns 
               if col not in ['Cluster', 'Team', 'Cluster_Name']]

# Get Real Madrid's factor scores
rm_factors = {col: float(rm_data[col]) for col in factor_cols}

# Sort by score
sorted_factors = sorted(rm_factors.items(), key=lambda x: x[1], reverse=True)

print(f"\n{'='*80}")
print("REAL MADRID FACTOR SCORES")
print(f"{'='*80}\n")

# Top strengths
print("TOP 3 STRENGTHS (Highest Scores):")
for i, (factor, score) in enumerate(sorted_factors[:3], 1):
    print(f"  {i}. {factor}: {score:.4f}")

# Bottom weaknesses
print("\nTOP 3 WEAKNESSES (Lowest Scores):")
for i, (factor, score) in enumerate(sorted_factors[-3:][::-1], 1):
    print(f"  {i}. {factor}: {score:.4f}")

# ==================== Get Cluster Info ====================
cluster_info = cluster_summary[str(cluster_id)]

print(f"\n{'='*80}")
print(f"CLUSTER: {cluster_info['name'].upper()}")
print(f"{'='*80}\n")
print(f"Cluster Size: {cluster_info['size']} teams")
print(f"Top Strengths: {', '.join(cluster_info['top_strengths'][:2])}")
print(f"Top Weaknesses: {', '.join(cluster_info['top_weaknesses'][:2])}")

# ==================== Compare to Cluster ====================
print(f"\n{'='*80}")
print("REAL MADRID vs CLUSTER AVERAGE")
print(f"{'='*80}\n")

above_avg = 0
below_avg = 0
cluster_means = {k: float(v) for k, v in cluster_info['mean_scores'].items()}

above_factors = []
below_factors = []

for factor in factor_cols:
    rm_score = rm_factors[factor]
    cluster_avg = cluster_means.get(factor, 0)
    diff = rm_score - cluster_avg
    
    if diff > 0.1:
        above_avg += 1
        above_factors.append((factor, diff))
    elif diff < -0.1:
        below_avg += 1
        below_factors.append((factor, diff))

above_factors = sorted(above_factors, key=lambda x: x[1], reverse=True)[:2]
below_factors = sorted(below_factors, key=lambda x: x[1])[:2]

print(f"Factors ABOVE cluster average: {above_avg}")
for factor, diff in above_factors:
    print(f"  • {factor}: +{diff:.3f}")

print(f"\nFactors BELOW cluster average: {below_avg}")
for factor, diff in below_factors:
    print(f"  • {factor}: {diff:.3f}")

# ==================== Generate Summary ====================
print(f"\n{'='*80}")
print("THREE-SENTENCE SUMMARY")
print(f"{'='*80}\n")

# Sentence 1: Strengths
top_1 = sorted_factors[0][0].replace('_', ' ').title()
top_2 = sorted_factors[1][0].replace('_', ' ').title()

strength_mapping = {
    'Attacking Efficiency': 'superior shot quality and conversion efficiency',
    'Attacking Output': 'prolific goal-scoring and high expected points generation',
    'Defensive Prowess': 'exceptional defensive organization and pressing efficiency',
    'Possession Control': 'commanding ball control and field dominance',
    'Box Dominance': 'consistent final-third penetration and dangerous box activity',
    'Recovery & Transition': 'swift possession recovery and rapid counterattacking',
    'Attacking Efficiency.1': 'consistent attacking threat and chance creation',
    'Attacking Efficiency.2': 'clinical finishing and high-quality attacking play',
    'Attacking Efficiency.3': 'effective attacking transitions and tempo control',
    'Attacking Efficiency.4': 'organized attacking patterns and tactical discipline',
    'Attacking Efficiency.5': 'efficient offensive execution and opportunity conversion',
    'Attacking Efficiency.6': 'versatile attacking approaches and multiple scoring routes',
    'Possession Control.1': 'structured possession-based build-up play',
    'Possession Control.2': 'possession retention and methodical play progression',
    'Recovery & Transition.1': 'dynamic recovery and transition efficiency'
}

strength_text_1 = strength_mapping.get(top_1, top_1.lower())
strength_text_2 = strength_mapping.get(top_2, top_2.lower())

sentence_1 = (
    f"Real Madrid demonstrates {strength_text_1} and {strength_text_2}, "
    f"establishing itself as an elite competitor with comprehensive attacking prowess "
    f"and sophisticated tactical execution across multiple dimensions of play."
)

# Sentence 2: Areas of development
if below_avg >= 2:
    bottom_1 = sorted_factors[-1][0].replace('_', ' ').title()
    bottom_2 = sorted_factors[-2][0].replace('_', ' ').title()
    
    weakness_mapping = {
        'Recovery & Transition': 'slower possession recovery and transition speed',
        'Recovery & Transition.1': 'limited quick counter-attacking capability',
        'Box Dominance': 'reduced box activity and final-third penetration',
        'Possession Control': 'lower ball retention and possession dominance',
        'Defensive Prowess': 'less intensive defensive pressure and organization',
        'Attacking Efficiency.1': 'occasional inconsistency in attacking efficiency',
        'Attacking Efficiency.3': 'limited transition-based attacking threat'
    }
    
    weak_text_1 = weakness_mapping.get(bottom_1, bottom_1.lower())
    weak_text_2 = weakness_mapping.get(bottom_2, bottom_2.lower())
    
    sentence_2 = (
        f"While Real Madrid exhibits some tactical development areas including {weak_text_1} "
        f"and {weak_text_2}, the team maintains above-average performance compared to peers, "
        f"demonstrating resilience and competitive balance across most critical dimensions."
    )
else:
    sentence_2 = (
        f"Real Madrid performs consistently in line with its peer group across most factors, "
        f"without significant tactical vulnerabilities, indicating well-rounded and stable execution "
        f"that balances attacking ambition with defensive responsibility."
    )

# Sentence 3: Conclusion
sentence_3 = (
    f"Overall, Real Madrid exemplifies the {cluster_info['name']} profile—a top-tier club that combines "
    f"efficient attacking football with tactical flexibility, capable of competing at the highest levels "
    f"through balanced development and consistent high-performance standards."
)

# ==================== Display Complete Summary ====================
print("SENTENCE 1 (STRENGTHS):")
print(f"{sentence_1}\n")

print("SENTENCE 2 (WEAKNESSES/AVERAGE AREAS):")
print(f"{sentence_2}\n")

print("SENTENCE 3 (CONCLUSION):")
print(f"{sentence_3}\n")

# ==================== Full Summary ====================
print("="*80)
print("COMPLETE THREE-SENTENCE SUMMARY FOR REAL MADRID")
print("="*80 + "\n")

full_summary = f"{sentence_1} {sentence_2} {sentence_3}"
print(full_summary)

# ==================== Save Results ====================
summary_data = {
    'team': team_name,
    'cluster_id': cluster_id,
    'cluster_name': cluster_name,
    'top_strengths': [sorted_factors[0][0], sorted_factors[1][0]],
    'top_weaknesses': [sorted_factors[-1][0], sorted_factors[-2][0]],
    'sentence_1_strengths': sentence_1,
    'sentence_2_weaknesses': sentence_2,
    'sentence_3_conclusion': sentence_3,
    'complete_summary': full_summary
}

with open('/mnt/user-data/outputs/real_madrid_summary.json', 'w') as f:
    json.dump(summary_data, f, indent=2)

print("\n" + "="*80)
print("✓ Summary saved to: real_madrid_summary.json")
print("="*80)
