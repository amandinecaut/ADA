# Cluster Analysis of Cat Breeds Based on Factor Scores

## Executive Summary

A K-means cluster analysis was performed on 62 cat breeds using the 4 factor scores derived from factor analysis. The analysis identified **4 distinct clusters** (silhouette score: 0.2729) that represent different cat breed archetypes based on size, lifespan, sociability, and maintenance requirements.

---

## Clustering Methodology

### Data & Features
- **Total Breeds Analyzed:** 62
- **Clustering Features:** 4 factor scores from previous factor analysis
  - Factor 1: Physical Size
  - Factor 2: Longevity & Lifespan
  - Factor 3: Sociability & Temperament
  - Factor 4: Maintenance & Grooming

### Cluster Determination
- **Method:** K-means clustering with elbow method and silhouette score analysis
- **Optimal Clusters:** 4 (highest silhouette score: 0.2729)
- **Validation:** Hierarchical clustering dendrogram confirms cluster separation

---

## The Four Clusters

### **Cluster 0: Robust & Reserved** (12 breeds, 19.4%)

**Overview:**
The Robust & Reserved cluster comprises 12 large cat breeds with long-lived lifespans and a moderate temperament. These are substantial felines that offer longevity and stability to their families while maintaining independent but balanced personalities.

**Strengths:**
These breeds excel in good playfulness and energy levels, combined with strong intelligence and trainability. Their exceptional longevity (averaging 14.4-19.2 years) makes them long-term companions, and their natural intelligence facilitates learning and interaction.

**Weaknesses:**
However, they require moderate to high grooming maintenance (averaging 4.4/5 on the grooming scale) and some shedding control, which demands consistent care routines from owners. Despite their moderate sociability, they may not be ideal for extremely family-focused households seeking highly affectionate breeds.

**Representative Characteristics:**
- Weight Range: 8.1-15.1 kg
- Lifespan: 14.4-19.2 years
- Family Friendly: 3.9/5
- Grooming: 4.4/5
- Intelligence: 4.2/5

**Key Breeds:** American Shorthair, American Wirehair, Australian Mist, Bombay, Brazilian Shorthair, European Shorthair, Kurilian Bobtail, Mekong Bobtail, Savannah, Snowshoe, Turkish Van

---

### **Cluster 1: Sophisticated Independents** (7 breeds, 11.3%)

**Overview:**
The Sophisticated Independents cluster comprises 7 medium-sized cat breeds with moderate lifespans and distinctly independent temperaments. These breeds embody an air of refinement and self-sufficiency, preferring solitude to constant interaction.

**Strengths:**
These breeds excel in their distinctive personalities and elegant demeanor, offering owners a sense of exclusivity and mystery. They require balanced grooming (averaging 2.4/5) and minimal shedding (averaging 3.7/5), making them ideal for owners who prefer low-maintenance companions with sophisticated charm.

**Weaknesses:**
However, they tend to struggle with limited suitability for homes with children, as their independent and aloof nature (sociability score: 1.624) means they may not engage warmly with younger family members. Their lower playfulness (averaging 2.9/5) and preference for solitude make them less ideal for households seeking highly interactive pets.

**Representative Characteristics:**
- Weight Range: 7.3-14.4 kg
- Lifespan: 10.4-16.0 years
- Family Friendly: 3.7/5
- Playfulness: 2.9/5 (least playful cluster)
- Children Friendly: 3.0/5
- Grooming: 2.4/5 (minimal grooming needs)

**Key Breeds:** British Longhair, British Shorthair, Chantilly-Tiffany, Manx, Nebelung, Persian, Russian Blue

---

### **Cluster 2: Playful Sprites** (35 breeds, 56.5%)

**Overview:**
The Playful Sprites cluster comprises 35 small cat breeds with moderate lifespans and balanced temperaments, representing the most diverse and popular cat breed group. These are energetic, engaging companions perfect for active families and multi-pet households.

**Strengths:**
These breeds excel in excellent playfulness and energy levels (averaging 4.3/5), combined with being highly family and child-friendly (4.6/5 and 4.5/5 respectively). Their small size makes them ideal apartment dwellers, their superior intelligence (4.2/5) enables easy training, and their excellent compatibility with other pets (4.0/5) makes them outstanding for blended pet families.

**Weaknesses:**
However, they require relatively consistent care with moderate grooming needs (averaging 3.5/5) and some shedding management. Despite their many strengths, some breeds within this cluster show lower general health ratings (averaging 3.0/5), requiring vigilant veterinary care to ensure optimal wellness.

**Representative Characteristics:**
- Weight Range: 6.4-11.7 kg (smallest cluster)
- Lifespan: 10.4-14.9 years
- Family Friendly: 4.6/5 (highest)
- Playfulness: 4.3/5 (highest)
- Children Friendly: 4.5/5 (highest)
- Other Pets Friendly: 4.0/5

**Key Breeds (35 total, including):** Abyssinian, Bengali Cats, Burmese, Cornish Rex, Devon Rex, Japanese Bobtail, Siamese Cat, Sphynx, Tonkinese, Turkish Angora, Somali, and many others

---

### **Cluster 3: Laid-back Giants** (8 breeds, 12.9%)

**Overview:**
The Laid-back Giants cluster comprises 8 large cat breeds with shorter lifespans but exceptional temperaments, combining impressive physical presence with gentle, social personalities. These magnificent cats offer dramatic presence without demanding constant attention.

**Strengths:**
These breeds excel in excellent playfulness and energy levels (averaging 4.3/5) while maintaining remarkable family-friendliness (4.4/5) and outstanding compatibility with children (4.4/5). Their naturally low shedding (averaging 2.3/5) and minimal grooming requirements (averaging 2.7/5) make them surprisingly easy to maintain despite their large size, and their superior general health (4.1/5) provides reliable wellness.

**Weaknesses:**
However, they tend to struggle with shorter lifespans compared to other large breeds (averaging 11-20 years, with 3 breeds having notably lower maximum ages around 17 years), meaning owners face earlier loss of their cherished companions. The combination of large size and shorter lifespan represents a significant trade-off that prospective owners should carefully consider.

**Representative Characteristics:**
- Weight Range: 9.0-24.0 kg (largest cluster on average)
- Lifespan: Shorter than Robust & Reserved cluster
- Family Friendly: 4.4/5
- Playfulness: 4.3/5
- Children Friendly: 4.4/5
- Shedding: 2.3/5 (low)
- Grooming: 2.7/5 (minimal)

**Key Breeds:** Aphrodite Giant, Chausie, Highlander, Maine Coon, Pixie-Bob, Ragdoll Cats, Siberian, York Chocolate

---

## Cluster Comparison Table

| Characteristic | Robust & Reserved | Sophisticated Independents | Playful Sprites | Laid-back Giants |
|---|---|---|---|---|
| **Size** | Large | Medium | Small | Large |
| **Number of Breeds** | 12 | 7 | 35 | 8 |
| **Percentage** | 19.4% | 11.3% | 56.5% | 12.9% |
| **Avg Weight (kg)** | 8.1-15.1 | 7.3-14.4 | 6.4-11.7 | 9.0-24.0 |
| **Avg Lifespan (years)** | 14.4-19.2 | 10.4-16.0 | 10.4-14.9 | 11-20 |
| **Playfulness** | 3.8/5 | 2.9/5 | 4.3/5 | 4.3/5 |
| **Family Friendly** | 3.9/5 | 3.7/5 | 4.6/5 | 4.4/5 |
| **Children Friendly** | 4.2/5 | 3.0/5 | 4.5/5 | 4.4/5 |
| **Intelligence** | 4.2/5 | 3.7/5 | 4.2/5 | 4.0/5 |
| **Shedding** | 2.3/5 | 3.7/5 | 2.1/5 | 2.3/5 |
| **Grooming** | 4.4/5 | 2.4/5 | 3.5/5 | 2.7/5 |
| **General Health** | 4.0/5 | 3.4/5 | 3.0/5 | 4.1/5 |

---

## Buyer's Guide: Which Cluster is Right for You?

### Choose **Robust & Reserved** if you:
- Want a large, impressive breed
- Value longevity and lifelong companionship (14-19 years)
- Are willing to invest in regular grooming
- Seek intelligent, trainable companions
- Have moderate family engagement preferences

### Choose **Sophisticated Independents** if you:
- Prefer low-maintenance grooming needs
- Appreciate independent, aloof personalities
- Want a medium-sized companion
- Are comfortable without constant affection
- Don't have young children in the household

### Choose **Playful Sprites** if you:
- Have an active, family-oriented household
- Want highly interactive, playful cats
- Have children or other pets
- Prefer small, apartment-friendly breeds
- Value trainability and intelligence

### Choose **Laid-back Giants** if you:
- Love large, impressive cats but want docile personalities
- Appreciate low-maintenance grooming
- Want good family and child compatibility
- Can accept shorter lifespans
- Seek dramatic presence without demanding care

---

## Clustering Insights

### Silhouette Analysis
- The optimal K=4 was selected based on the highest silhouette score (0.2729)
- The silhouette analysis shows moderate cluster cohesion, indicating that while the clusters are distinct, there is natural variation within breeds

### Hierarchical Clustering Dendrogram
- The dendrogram reveals four main branch clusters at the highest level
- The Playful Sprites cluster (35 breeds) shows the most internal diversity, reflecting its larger size and varied breed characteristics
- Smaller clusters (Sophisticated Independents, Laid-back Giants) show tighter grouping, indicating more homogeneous characteristics

### Factor Loadings by Cluster
- **Cluster 0** is defined primarily by high longevity (Factor 2: 1.388) and large size (Factor 1: 0.672)
- **Cluster 1** is uniquely characterized by high independence/aloofness (Factor 3: 1.624)
- **Cluster 2** is the most balanced across factors, with moderate scores in all dimensions
- **Cluster 3** is distinguished by extremely high size (Factor 1: 1.399) and high shedding/low grooming needs (Factor 4: 0.736)

---

## Practical Applications

### For Cat Breeders
- Understand market segments and breed positioning
- Identify gaps in the cluster landscape
- Develop breeding strategies aligned with cluster characteristics

### For Prospective Cat Owners
- Use the clusters to narrow down breed options matching lifestyle
- Compare within clusters for minor preference differences
- Understand trade-offs (e.g., size vs. lifespan in Laid-back Giants)

### For Veterinarians
- Tailor health management protocols by cluster
- Understand breed-specific health trends within clusters
- Provide cluster-specific preventive care recommendations

### For Shelter Operations
- Categorize adoptable cats by cluster for better matching
- Provide cluster-based guidance to adopters
- Develop cluster-specific behavioral programs

---

## Technical Notes

- **Clustering Algorithm:** K-means with k=4
- **Validation Metrics:** Elbow method, Silhouette score (0.2729)
- **Hierarchical Validation:** Ward linkage dendrogram
- **Data Preprocessing:** Factor scores were used directly (previously standardized)
- **Visualization:** 2D projections using pairs of factors with cluster centroids marked

---

## Files Generated

1. `cluster_analysis.py` - Complete Python implementation
2. `cluster_analysis_results.png` - Comprehensive visualization with 5 plots
3. `cluster_profiles_heatmap.png` - Mean factor scores by cluster
4. `breed_cluster_assignments.csv` - Full breed-to-cluster mapping with scores
5. `cluster_summary.csv` - Summary statistics for each cluster
6. `cluster_analysis_report.txt` - Detailed text report
