# Copilot Instructions for ADA Pipeline

## Project Overview
ADA (Automatic Data Analysis) is a Streamlit-based data exploration and clustering application that transforms raw datasets into interpretable insights through Factor Analysis (FA/FAMD), clustering, and AI-powered explanations.

**Key Flow:** Data Upload → Factor Analysis → Component Labeling (via LLM) → Clustering → Interactive Visualization & Chat

## Architecture Patterns

### Session State Management
All persistent data flows through Streamlit's `st.session_state`. Key patterns:
- Use `default_values` dict in `app_utilities.py` to initialize state
- Always check `if key not in st.session_state` before accessing
- Use `clear_session_state(skip=[])` to reset workflow while preserving file/map uploads
- Example: `st.session_state.FA_component_dict` stores factor metadata after analysis

### Multi-Strategy Factor Analysis
`factor_analysis.py` uses the **Strategy Pattern** with two implementations:
- **ContinuousFAStrategy** (`ContinuousFAStrategy`): Pure numerical data using scikit-learn's `FactorAnalysis`
- **FAMDStrategy** (`FAMDStrategy`): Mixed data types using `prince.FAMD`
- The app auto-selects via `select_strategy(df)` based on column types
- Always extract components differently: `model.components_` (FA) vs `model.column_coordinates_` (FAMD)

### LLM Integration Pattern
`wordalisation.py` provides abstract base classes for LLM tasks with strategy injection:
- **FALabel**: Generates semantic labels for factors from component loadings
- **QandAWordalisation**: Creates Q&A pairs explaining factor implications
- **ClusterWordalisation**: Names/describes clusters based on centroid composition
- All use "tell_it" methods to inject context (e.g., `tell_it_what_data_to_use()`)
- Supports both OpenAI and Google Gemini via `_config` from `.streamlit/secrets.toml`

### Data Pipeline Steps
1. **Load Tab (`app.py` lines 44-115)**: CSV detection (auto-delimiter & encoding), optional column mapping
2. **Analysis Tab**: Execute `perform_FA()` → generates `FA_component_dict` with labels & loadings
3. **Clustering Tab**: `perform_clustering()` → KMeans on FA scores → store labels/centroids
4. **View Tab**: Interactive scatter plots via Plotly + entity-specific chat context

## Critical Conventions

### Component Dictionary Structure
`FA_component_dict[f"Factor {i+1}"]` always contains:
```python
{
    "label": "semantic description from LLM",
    "top": [list of high-loading features],
    "values_top": [rounded loadings],
    "bottom": [low-loading features],
    "values_bottom": [negative loadings]
}
```

### Column Mapping
User-provided maps (JSON or Excel with "Key"/"Value" cols) are stored in `st.session_state.col_mapping`. 
**Always apply mapping** when displaying features to users (see `app_utilities.py` lines 475-477).

### Entity Metadata
- `st.session_state.entity_id`: User-defined entity type (e.g., "person", "dog")
- `st.session_state.article`: Auto-computed "a" vs "an" via `choose_article()`
- Used in LLM prompts for grammatical correctness

### DataFrames Used
- `df_full`: Raw loaded data
- `df_filtered`: Cleaned (no NaN), features selected, entity as index
- `df`: Z-score normalized factor scores (used for clustering/viz)
- `df_original`: Copy of `df` before clustering modifications

## File Organization

| Module | Purpose |
|--------|---------|
| `app.py` | Streamlit UI with 4 tabs; orchestrates workflow |
| `app_utilities.py` | Data loading, FA workflow, clustering, QA generation |
| `factor_analysis.py` | FA/FAMD strategy classes, auto-selection logic |
| `wordalisation.py` | LLM integrations for labeling/explanation (1438 lines) |
| `chat.py` | Entity-specific chat context + message history |
| `clustering.py` | KMeans + cluster naming/description |
| `visualisation_utilities.py` | Plotly figures for distribution/cluster/3D views |
| `embeddings.py` | Embedding generation (referenced but not heavily used) |

## External Dependencies & Configuration

### `.streamlit/secrets.toml` (Required)
```toml
[settings]
USE_GEMINI = true  # Toggle between Gemini and OpenAI

[services.gemini]
GEMINI_API_KEY = "..."
GEMINI_CHAT_MODEL = "gemini-2.5-flash"  # Configurable
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"

[services.gpt]
GPT_BASE = "..."
GPT_KEY = "..."
GPT_ENGINE = "gpt-4o-mini"
```

### Key Dependencies
- **Streamlit**: UI framework with session state management
- **scikit-learn**: FA, KMeans, StandardScaler
- **prince**: FAMD for mixed-type data
- **plotly**: Interactive visualizations
- **google-generativeai / openai**: LLM APIs (both supported)
- **pandas / numpy**: Data manipulation

## Workflow Checklist for New Features

1. **Data flows**: Ensure updates to `st.session_state` match `default_values` types
2. **Strategy check**: Does new analysis need FA/FAMD handling? See `perform_FA()` branching at line 328
3. **Column mapping**: Apply `col_mapping` when showing features to end users
4. **LLM calls**: Use `Wordalisation` subclasses; inject context via `tell_it_*` methods
5. **Index integrity**: Preserve DataFrame index (set to entity names in `update_df()`)
6. **Z-score normalization**: FA scores are normalized; clustering & viz use this normalized version

## Common Gotchas

- **Missing `.streamlit/secrets.toml`**: App will fail silently on LLM calls
- **NaN handling**: `update_df()` drops rows with ANY NaN; validate data in load tab
- **Component shapes**: FA outputs `(n_features, n_factors)` but FAMD returns `(n_factors, n_features)` → see branching at line 328
- **Cluster color map**: Uses Plotly's `Set1` palette; limited to ~9 clusters before color repetition
- **Streamlit reruns**: All code reruns on widget change; avoid expensive operations in loops without caching

## Running Locally
```bash
streamlit run src/app.py
```
Requires Python 3.10+ and `.streamlit/secrets.toml` with valid API keys.
