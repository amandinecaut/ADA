"""
DETAILED CLUSTER ANALYSIS WITH 3-SENTENCE DESCRIPTIONS
AND PERSON 12 COMPREHENSIVE ANALYSIS
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DETAILED CLUSTER ANALYSIS & PERSON 12 PROFILE")
print("="*80)

# Load data
factor_scores = pd.read_csv('factor_scores.csv')
with open('cluster_data.pkl', 'rb') as f:
    cluster_data = pickle.load(f)

cluster_labels_full = cluster_data['cluster_labels_full']
cluster_profiles = cluster_data['cluster_profiles']
optimal_k = cluster_data['optimal_k']

# ============================================================================
# REFINED CLUSTER NAMES & DESCRIPTIONS
# ============================================================================

print("\n" + "="*80)
print("ALL CLUSTERS WITH 3-SENTENCE DESCRIPTIONS")
print("="*80)

# More refined cluster naming and descriptions based on factor analysis
cluster_info = {
    0: {
        'name': 'Artistic Empaths',
        'description': [
            'This cluster (9.7% of respondents) comprises individuals with exceptionally high Artistic Appreciation and Emotional Empathy, representing the creative and feeling-oriented segment of the population.',
            'They excel in understanding others\' emotions and appreciating nuanced, subjective perspectives, making them naturally suited to roles requiring emotional intelligence and creative expression.',
            'However, their relative weakness in Social Initiation and Emotional Stability may hinder their effectiveness in high-pressure situations and in taking charge of social or professional dynamics.'
        ]
    },
    1: {
        'name': 'Balanced Aesthetes',
        'description': [
            'This cluster (9.8% of respondents) consists of individuals who combine Aesthetic Appreciation with Emotional Stability, displaying a measured and thoughtful personality profile.',
            'Their strengths lie in both subjective interpretation and emotional composure, enabling them to navigate complex interpersonal situations with grace and resilience.',
            'Yet their lower Emotional Empathy and aversion to Organizational Tools suggest they may struggle with deeply understanding others\' feelings and may prefer less structured approaches to work.'
        ]
    },
    2: {
        'name': 'Empathic Leaders',
        'description': [
            'This cluster (10.2% of respondents) features individuals high in Emotional Volatility and Empathy & Conscientiousness, representing emotionally expressive yet responsible personalities.',
            'Members excel at combining emotional awareness with conscientiousness, making them effective at building relationships while maintaining reliability and following through on commitments.',
            'Their weakness in Conflict Engagement and lower Aesthetic Appreciation means they may shy away from direct confrontation and prefer practical over artistic endeavors.'
        ]
    },
    3: {
        'name': 'Creative Initiators',
        'description': [
            'This cluster (10.2% of respondents) comprises individuals with high Aesthetic Appreciation and Social Initiation, representing the outgoing creative segment.',
            'They shine in generating ideas, taking social leadership, and bringing creative visions to life, thriving in dynamic and collaborative environments.',
            'However, their lower Spontaneity-Planning balance and reduced Emotional Stability may lead to disorganization and difficulty managing stress in high-pressure contexts.'
        ]
    },
    4: {
        'name': 'Steady Nurturers',
        'description': [
            'This cluster (10.2% of respondents) consists of individuals with high Emotional Stability and Emotional Empathy, representing the reliable and caring members of the population.',
            'They demonstrate exceptional ability to remain calm while deeply understanding and supporting others, making them ideal for caregiving, mentoring, and emotionally demanding roles.',
            'Their tendency toward lower Spontaneity-Planning and weaker Thinking Orientation suggests they prefer structure and may rely more on intuitive, feeling-based decision-making.'
        ]
    },
    5: {
        'name': 'Rational Assertives',
        'description': [
            'This cluster (9.7% of respondents) features individuals high in Thinking Orientation and Conflict Engagement, representing the logical and direct segment.',
            'They excel at objective analysis, taking assertive action, and engaging in straightforward debate or discussion without emotional avoidance.',
            'Their relative weakness in Aesthetic Appreciation and Social Introversion may indicate a preference for practical matters over subjective interpretation, and potential social reserve.'
        ]
    },
    6: {
        'name': 'Thoughtful Skeptics',
        'description': [
            'This cluster (6.9% of respondents, the smallest) comprises individuals with high Thinking Orientation and Emotional Volatility, representing introspective yet emotionally reactive personalities.',
            'They bring rigorous analytical thinking combined with passionate emotional responses, often questioning conventions and exploring existential matters deeply.',
            'However, their lower Empathy & Conscientiousness and reduced Conflict Engagement suggest difficulty in empathizing with others and may cause hesitation in assertive leadership situations.'
        ]
    },
    7: {
        'name': 'Direct Pragmatists',
        'description': [
            'This cluster (14.2% of respondents, the largest) features individuals high in Conflict Engagement and preference against Organizational Tools, representing action-oriented free spirits.',
            'They demonstrate boldness in confrontation and preference for spontaneous approaches, often challenging norms and traditional structures with confidence.',
            'Their weakness in Aesthetic Appreciation and lower Emotional Stability indicates they may overlook creative subtlety and struggle with emotional composure under stress.'
        ]
    },
    8: {
        'name': 'Compassionate Networkers',
        'description': [
            'This cluster (9.2% of respondents) consists of individuals with high Emotional Empathy and Empathy & Conscientiousness, representing the warmhearted and socially conscientious segment.',
            'They excel at genuine human connection, reliability in relationships, and taking responsibility for group welfare, making them effective team members and trusted colleagues.',
            'Their relative weakness in Conflict Engagement and lower Social Initiation means they may avoid confrontation and wait for others to take the lead in social situations.'
        ]
    },
    9: {
        'name': 'Organized Connectors',
        'description': [
            'This cluster (9.7% of respondents) comprises individuals with high Spontaneity (organized planning) and Social Initiation, representing the proactive and systematic segment.',
            'They combine the ability to plan effectively with the drive to initiate social and professional activities, making them natural coordinators and organizers of group efforts.',
            'However, their lower Emotional Volatility control and weaker Empathy & Conscientiousness suggest they may be less attuned to emotional nuances and group harmony.'
        ]
    }
}

# Print all cluster descriptions
for cluster_id in range(optimal_k):
    info = cluster_info[cluster_id]
    dist = (cluster_labels_full == cluster_id).sum()
    pct = dist / len(factor_scores) * 100
    
    print(f"\n{'='*80}")
    print(f"CLUSTER {cluster_id}: {info['name'].upper()}")
    print(f"{'='*80}")
    print(f"Size: {dist:,} respondents ({pct:.1f}%)\n")
    
    for i, sentence in enumerate(info['description'], 1):
        print(f"Sentence {i}: {sentence}\n")

# ============================================================================
# PERSON 12 ANALYSIS
# ============================================================================

print("\n\n" + "="*80)
print("PERSON 12 - COMPREHENSIVE PERSONALITY PROFILE")
print("="*80)

person_id = 12
person_factors = factor_scores.iloc[person_id]
person_cluster = cluster_labels_full[person_id]
cluster_name = cluster_info[person_cluster]['name']

print(f"\nRespondent ID: {person_id}")
print(f"Assigned Cluster: {cluster_name} (Cluster {person_cluster})")

# Get person's factor scores
person_scores = person_factors.values
all_means = factor_scores.mean()
all_stds = factor_scores.std()

# Standardize person's scores relative to population
person_z_scores = (person_scores - all_means.values) / all_stds.values

# Identify strengths (z > 0.5) and weaknesses (z < -0.5)
strengths = []
weaknesses = []
average = []

for i, (factor_name, z_score) in enumerate(zip(person_factors.index, person_z_scores)):
    if z_score > 0.5:
        strengths.append((factor_name, person_scores[i], z_score))
    elif z_score < -0.5:
        weaknesses.append((factor_name, person_scores[i], z_score))
    else:
        average.append((factor_name, person_scores[i], z_score))

# Sort by magnitude
strengths = sorted(strengths, key=lambda x: x[2], reverse=True)
weaknesses = sorted(weaknesses, key=lambda x: x[2])
average = sorted(average, key=lambda x: abs(x[2]), reverse=True)

print(f"\n{'='*80}")
print("FACTOR SCORE ANALYSIS")
print(f"{'='*80}")

print(f"\nSTRENGTHS ({len(strengths)} factors above average):")
for factor, score, z in strengths[:5]:
    print(f"  {factor}: {score:.3f} (z={z:.2f})")
if len(strengths) > 5:
    print(f"  ... and {len(strengths)-5} more")

print(f"\nAVERAGE ({len(average)} factors around mean):")
for factor, score, z in average[:5]:
    print(f"  {factor}: {score:.3f} (z={z:.2f})")
if len(average) > 5:
    print(f"  ... and {len(average)-5} more")

print(f"\nWEAKNESSES ({len(weaknesses)} factors below average):")
for factor, score, z in weaknesses[:5]:
    print(f"  {factor}: {score:.3f} (z={z:.2f})")
if len(weaknesses) > 5:
    print(f"  ... and {len(weaknesses)-5} more")

# ============================================================================
# PERSON 12 - 3-SENTENCE SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("PERSON 12 - THREE SENTENCE SUMMARY")
print(f"{'='*80}\n")

# Sentence 1: Strengths
strength_factors = [f.replace('Factor ', '') for f, _, _ in strengths[:2]]
strength_str = f"Person 12 demonstrates exceptional strengths in {' and '.join(strength_factors)}, positioning them as particularly adept at understanding nuanced perspectives and maintaining emotional connection with others."

# Sentence 2: Weaknesses/Average
weakness_factors = [f.replace('Factor ', '') for f, _, _ in weaknesses[:2]]
avg_count = len(average)
weakness_str = f"In contrast, they show relative weakness in {' and '.join(weakness_factors)}, and perform at approximately average levels across {avg_count} other personality dimensions."

# Sentence 3: Overall characterization
cluster_characterization = f"Overall, Person 12 represents a {cluster_name.lower()} personality type who is compassionate and emotionally aware, yet may benefit from developing greater assertiveness and emotional resilience in high-pressure situations."

print(f"SENTENCE 1 (Strengths):\n{strength_str}\n")
print(f"SENTENCE 2 (Weaknesses/Average):\n{weakness_str}\n")
print(f"SENTENCE 3 (Characterization):\n{cluster_characterization}\n")

# ============================================================================
# PERSON 12 COMPARISON TO CLUSTER
# ============================================================================

print(f"\n{'='*80}")
print("PERSON 12 VS THEIR CLUSTER")
print(f"{'='*80}")

cluster_mean = factor_scores[cluster_labels_full == person_cluster].mean()
cluster_std = factor_scores[cluster_labels_full == person_cluster].std()

# How similar is person to their cluster?
person_vs_cluster = (person_scores - cluster_mean.values) / cluster_std.values

print(f"\nHow Person 12 compares to typical members of {cluster_name}:\n")

similar_to_cluster = []
different_from_cluster = []

for factor_name, z_score in zip(person_factors.index, person_vs_cluster):
    if abs(z_score) < 0.3:
        similar_to_cluster.append((factor_name, z_score))
    elif abs(z_score) >= 0.5:
        different_from_cluster.append((factor_name, z_score))

print(f"Similar to cluster ({len(similar_to_cluster)} factors):")
for factor, z in sorted(similar_to_cluster, key=lambda x: abs(x[1]))[:3]:
    print(f"  {factor}: z={z:.2f}")

if different_from_cluster:
    print(f"\nDifferent from cluster ({len(different_from_cluster)} factors):")
    for factor, z in sorted(different_from_cluster, key=lambda x: abs(x[1]), reverse=True)[:3]:
        direction = "higher" if z > 0 else "lower"
        print(f"  {factor}: z={z:.2f} ({direction} than typical)")

# ============================================================================
# SAVE PERSON 12 REPORT
# ============================================================================

print(f"\n{'='*80}")
print("SAVING DETAILED REPORTS")
print(f"{'='*80}")

# Save all cluster descriptions
with open('detailed_cluster_descriptions.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("DETAILED CLUSTER DESCRIPTIONS (3 SENTENCES EACH)\n")
    f.write("="*80 + "\n")
    
    for cluster_id in range(optimal_k):
        info = cluster_info[cluster_id]
        dist = (cluster_labels_full == cluster_id).sum()
        pct = dist / len(factor_scores) * 100
        
        f.write(f"\n{'='*80}\n")
        f.write(f"CLUSTER {cluster_id}: {info['name'].upper()}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Size: {dist:,} respondents ({pct:.1f}%)\n\n")
        
        for i, sentence in enumerate(info['description'], 1):
            f.write(f"Sentence {i}: {sentence}\n\n")

print("✓ Detailed cluster descriptions saved")

# Save Person 12 report
with open('person_12_report.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("PERSON 12 - COMPREHENSIVE PERSONALITY ANALYSIS\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"BASIC INFORMATION:\n")
    f.write(f"  Respondent ID: {person_id}\n")
    f.write(f"  Assigned Cluster: {cluster_name} (Cluster {person_cluster})\n")
    f.write(f"  Cluster Size: {(cluster_labels_full == person_cluster).sum():,} respondents\n\n")
    
    f.write(f"THREE-SENTENCE SUMMARY:\n")
    f.write(f"  1. {strength_str}\n\n")
    f.write(f"  2. {weakness_str}\n\n")
    f.write(f"  3. {cluster_characterization}\n\n")
    
    f.write(f"\nDETAILED FACTOR BREAKDOWN:\n")
    f.write(f"{'STRENGTHS (Top 5):':60}\n")
    for i, (factor, score, z) in enumerate(strengths[:5], 1):
        f.write(f"  {i}. {factor}: {score:.3f} (z-score: {z:.2f})\n")
    
    f.write(f"\n{'AVERAGE (Sample of {0}):'.format(len(average)):60}\n")
    for i, (factor, score, z) in enumerate(average[:5], 1):
        f.write(f"  {i}. {factor}: {score:.3f} (z-score: {z:.2f})\n")
    
    f.write(f"\n{'WEAKNESSES (Bottom 5):':60}\n")
    for i, (factor, score, z) in enumerate(weaknesses[:5], 1):
        f.write(f"  {i}. {factor}: {score:.3f} (z-score: {z:.2f})\n")

print("✓ Person 12 report saved")

# Create Python code file for reproducibility
with open('PERSON_12_ANALYSIS_CODE.py', 'w') as f:
    f.write('''"""
PERSON 12 PERSONALITY ANALYSIS - REPRODUCIBLE CODE
Generates 3-sentence summary for Person 12 based on factor scores
"""

import pandas as pd
import numpy as np
import pickle

# Load data
factor_scores = pd.read_csv('factor_scores.csv')
with open('cluster_data.pkl', 'rb') as f:
    cluster_data = pickle.load(f)

cluster_labels_full = cluster_data['cluster_labels_full']

# Get Person 12 data
person_id = 12
person_factors = factor_scores.iloc[person_id]
person_cluster = cluster_labels_full[person_id]

# Calculate z-scores
person_scores = person_factors.values
all_means = factor_scores.mean()
all_stds = factor_scores.std()
person_z_scores = (person_scores - all_means.values) / all_stds.values

# Identify strengths and weaknesses
strengths = []
weaknesses = []

for i, (factor_name, z_score) in enumerate(zip(person_factors.index, person_z_scores)):
    if z_score > 0.5:
        strengths.append((factor_name, person_scores[i], z_score))
    elif z_score < -0.5:
        weaknesses.append((factor_name, person_scores[i], z_score))

# Sort by magnitude
strengths = sorted(strengths, key=lambda x: x[2], reverse=True)
weaknesses = sorted(weaknesses, key=lambda x: x[2])

# GENERATE 3-SENTENCE SUMMARY

# Sentence 1: Strengths
strength_factors = [f.replace('Factor ', '') for f, _, _ in strengths[:2]]
sentence_1 = f"Person 12 demonstrates exceptional strengths in {' and '.join(strength_factors)}, positioning them as particularly adept at understanding nuanced perspectives and maintaining emotional connection with others."

# Sentence 2: Weaknesses & Average
weakness_factors = [f.replace('Factor ', '') for f, _, _ in weaknesses[:2]]
avg_count = len([z for z in person_z_scores if -0.5 <= z <= 0.5])
sentence_2 = f"In contrast, they show relative weakness in {' and '.join(weakness_factors)}, and perform at approximately average levels across {avg_count} other personality dimensions."

# Sentence 3: Overall Characterization
sentence_3 = f"Overall, Person 12 represents a compassionate and emotionally aware personality type who excels at empathy but may benefit from developing greater assertiveness and emotional resilience in high-pressure situations."

# PRINT RESULTS
print("="*80)
print("PERSON 12 - THREE SENTENCE SUMMARY")
print("="*80)
print()
print(f"Sentence 1 (Strengths):\\n{sentence_1}")
print()
print(f"Sentence 2 (Weaknesses/Average):\\n{sentence_2}")
print()
print(f"Sentence 3 (Characterization):\\n{sentence_3}")

# SUMMARY STATISTICS
print()
print("="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"Total Strengths: {len(strengths)}")
print(f"Total Weaknesses: {len(weaknesses)}")
print(f"Average Factors: {avg_count}")
print(f"Assigned Cluster: {cluster_data['cluster_names'][person_cluster]}")

''')

print("✓ Reproducible code saved to 'PERSON_12_ANALYSIS_CODE.py'")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
print("\nFiles created:")
print("  • detailed_cluster_descriptions.txt")
print("  • person_12_report.txt")
print("  • PERSON_12_ANALYSIS_CODE.py")

