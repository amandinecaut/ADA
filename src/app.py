from visualisation_utilities import (
    Visualisation,
    ClusterVisualisation,
    ClusterVisualisation3D,
    DistributionPlot,
)
from wordalisation import CreateWordalisation
from clustering import Cluster
from google.generativeai import GenerationConfig
import google.generativeai as genai
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import json

import app_utilities
from app_utilities import *

from embeddings import Embeddings, embed
from factor_analysis import select_strategy, get_kaiser_criterion, HornParallelAnalysis, get_famd_metrics, get_kaiser_famd, get_strategy_description


st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Sidebar configuration
with st.sidebar:
    st.markdown("### Debug Options")
    show_gpt_calls = st.checkbox("Show GPT calls", value=True, help="Display LLM prompts and messages for debugging")
    if "show_gpt_calls" not in st.session_state:
        st.session_state.show_gpt_calls = show_gpt_calls
    else:
        st.session_state.show_gpt_calls = show_gpt_calls

default_factor_nb, default_threshold, default_max_components, default_num_clusters = (
    get_defaults()
)

for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value
    


# Add and app header
st.title("ADA Pipeline")

tab1, tab2, tab3, tab4 = st.tabs(["Load", "Factors", "Clusters", "Describe"])

FA_done = False

# Load Data
with tab1:
    col_select, col_load, col_content = st.columns([2, 1, 3])
    
    with col_select:
        data_options = ["Select a Dataset"] + list(DATA_PATHS.keys()) + ["Upload Custom Data"]
        st.selectbox(
            "Choose data source:",
            options=data_options,
            key="demo_dataset_choice", 
            index=1
        )
    
    with col_load:
        st.markdown("<br>", unsafe_allow_html=True)  # Align with selectbox
        if st.button("Load", use_container_width=True, type="primary"):
            if st.session_state.demo_dataset_choice not in ["Select a Dataset", "Upload Custom Data"]:
                set_default_data_callback()
    
    # Show different controls based on data source selection
    is_custom_upload = st.session_state.demo_dataset_choice == "Upload Custom Data"
    is_demo_selected = st.session_state.demo_dataset_choice != "Select a Dataset" and not is_custom_upload
    
    if is_custom_upload:
        with col_content:
            col_upload1, col_upload2, col_upload3 = st.columns(3)
            
            with col_upload1:
                st.file_uploader(
                    "CSV file",
                    type=["csv"],
                    key="file",
                    on_change=load_new_data,
                )
            
            with col_upload2:
                st.file_uploader(
                    "Column mapping (optional)",
                    type=["json", "xlsx", "xls"],
                    key="map",
                    on_change=load_map,
                )
            
            with col_upload3:
                user_input = st.text_input("Entity name:", key="entity_id")
                st.session_state.article = choose_article(st.session_state.entity_id)
    
    # Data Information Section

    if "df_full" not in st.session_state:
        
        if is_custom_upload:
            st.markdown(
                "**Note:** The uploaded dataset must be a **numerical DataFrame** with only numeric columns."
            )

            col_img1, col_img2, col_img3 = st.columns(3)
            with col_img1:
                st.image(
                    "./data/example_dataframe.png",
                    caption="Example of a valid numerical DataFrame",
                    use_column_width=True,
                )
            with col_img2:
                st.image(
                    "./data/example_json.png", 
                    caption="Example of a valid json file", 
                    use_column_width=True
                )
            with col_img3:
                st.markdown(
                    "The Excel file (`.xlsx` or `.xls`) must have:\n"
                    "- A first column named **`Key`**\n"
                    "- A second column named **`Value`**"
                )
                st.image(
                    "./data/example_xlsx.png", 
                    caption="Example of a valid xlsx file", 
                    use_column_width=True
                )

    else:
        if st.session_state.get("data_loading", False):
            st.info("Loading data, please wait...")
        else:
            # For demo datasets, show simplified view
            if is_demo_selected:
                # show st.session_state.df_filtered with the index column blue
                st.dataframe(st.session_state.df_filtered, use_container_width=True)

                expander_map = st.expander("Column mapping")
                expander_map.write(st.session_state.col_mapping)
                
                st.session_state.tab1_done = True

                col_name = st.session_state.get("col_name")
                option_row = st.session_state.df_filtered.index.to_list()
                if col_name is None:
                    option_labels = [f"{st.session_state.entity_id.capitalize()} №{i}" for i, _ in enumerate(option_row)]
                else: 
                    option_labels = st.session_state.df_full.loc[st.session_state.df_filtered.index, col_name] 
                label_to_value = dict(zip(option_labels, option_row))
            
                st.session_state["label_to_value"] = label_to_value
            
            # For custom uploads, show full configuration
            elif is_custom_upload:
                expander_sample = st.expander("Sample of the data")
                expander_sample.write(st.session_state.df_full.sample(5))

                cols = ["Index"] + st.session_state.df_full.columns.to_list()
                # drop down "select entity", default to "Index"
                entity = st.selectbox(
                    "Select column to index data",
                    cols,
                    index=0,
                    key="entity_col",
                    on_change=update_df,
                )

                if st.session_state.col_mapping != {}:
                    default_ignore = [
                        c
                        for c in st.session_state.df_full.columns.to_list()
                        if c not in st.session_state.col_mapping.keys()
                        and c != st.session_state.entity_col
                    ]
                else:
                    default_ignore = []

                if "ignore_cols" not in st.session_state:
                    update_df(default_ignore)

                # add check box to ignore certain columns
                ignore_cols = st.multiselect(
                    label="Ignore columns",
                    options=st.session_state.df_full.columns.to_list(),
                    default=default_ignore,
                    on_change=update_df,
                    key="ignore_cols",
                )

                # display warning if there are rows with NaN
                if (
                    st.session_state.df_full[st.session_state.features]
                    .isnull()
                    .any(axis=1)
                    .sum()
                    > 0
                ):
                    st.warning(
                        "There are rows containing NaN, these will be dropped."
                    )

                expander_nan = st.expander("Rows containing NaN")
                if st.session_state.entity_col == "Index":
                    expander_nan.write(
                        st.session_state.df_full[
                            st.session_state.df_full[st.session_state.features]
                            .isnull()
                            .any(axis=1)
                        ][st.session_state.features]
                    )
                else:
                    expander_nan.write(
                        st.session_state.df_full[
                            st.session_state.df_full[
                                [st.session_state.entity_col] + st.session_state.features
                            ]
                            .isnull()
                            .any(axis=1)
                        ][[st.session_state.entity_col] + st.session_state.features]
                    )

                # display warning if data is empty after dropping NaN
                if st.session_state.df_filtered.shape[0] == 0:
                    st.warning(
                        "Data is empty. Select a different column to index data."
                    )
                
                # show st.session_state.df_filtered with the index column blue
                update_df(st.session_state.ignore_cols)
                st.dataframe(st.session_state.df_filtered, use_container_width=True)

                expander_map = st.expander("Column mapping")
                expander_map.write(st.session_state.col_mapping)

                if "entity_radio" not in st.session_state:
                    st.session_state.entity_radio = "No"
                    
                entity_name_radio = st.radio(
                    f"Does your dataset contain a column for the names?",
                    options=["Yes", "No"],
                    key="entity_radio",
                )

                if entity_name_radio == "No":
                    st.session_state["col_name"] = None
                    
                elif entity_name_radio == "Yes":
                    if st.session_state.ignore_cols:
                        selected_from_ignore = st.selectbox(
                             f"Pick the column to use for the {st.session_state.entity_id} names",
                             options = st.session_state.ignore_cols,
                             key="selected_from_ignore",
                        )
                       
                        # store it in session_state
                        st.session_state["col_name"] = selected_from_ignore
                        expander_col_name = st.expander(f"Show the column used for the {st.session_state.entity_id} names")
                        expander_col_name.write(st.session_state.df_full.loc[st.session_state.df_filtered.index, [selected_from_ignore]])
                    else:
                        # Handle the case where the user selected "Yes" but there are no ignored columns
                        st.warning("To select an entity name column, it must be in the 'Ignore columns' list. Please select a column to ignore first.")

                st.session_state.tab1_done = True

                col_name = st.session_state.get("col_name")
                option_row = st.session_state.df_filtered.index.to_list()
                if col_name is None:
                    option_labels = [f"{st.session_state.entity_id.capitalize()} №{i}" for i, _ in enumerate(option_row)]
                else: 
                    option_labels = st.session_state.df_full.loc[st.session_state.df_filtered.index, selected_from_ignore] 
                label_to_value = dict(zip(option_labels, option_row))
              
                st.session_state["label_to_value"] = label_to_value

# "Analysis Tools"
with tab2:
    if not st.session_state.get("tab1_done", False):
        st.warning("You must load your data first!")
    elif "df_full" not in st.session_state:
        st.markdown("Load data to perform Factor Analysis")
    elif len(st.session_state.df_filtered) < 10:
        st.markdown("Not enough data to perform Factor Analysis")
    else:
        # Factor analysis choice
        data_to_analyse = st.session_state.df_filtered.loc[:, st.session_state.features]
        strategy_name = select_strategy(data_to_analyse).name
        st.session_state.strategy_name = strategy_name

        # Ask user if they want to automatically find optimal number of factors
        factor_auto = st.radio(
            "Select factor determination method:",
            ("Automatic (recommended)", "Manual selection"),
            index=0,
            help="Automatic optimization uses statistical tests to find the best fit for your data."
        )

        if factor_auto == "Automatic (recommended)":
           
            if strategy_name in "FA":
                with st.spinner("Calculating optimal factors..."):
                    kaiser_number = get_kaiser_criterion()
                    Horn = HornParallelAnalysis()
              
                # Using metrics for a more "Dashboard" feel
                m_col1, m_col2 = st.columns(2)
                m_col1.metric("Kaiser Criterion", f"{kaiser_number} Factors", help="A standard approach based on eigenvalues > 1.")
                m_col2.metric("Parallel Analysis", f"{Horn} Factors", help="A more robust statistical simulation method.")

                # Action Buttons
                col1, col2 = st.columns(2)
                if col1.button(f"Use Kaiser Criterion ({kaiser_number})", use_container_width=True, help="Eigenvalues > 1", type="primary"):
                    st.session_state.factor_nb = kaiser_number
                    perform_FA()

                if col2.button(f"Use Parallel Analysis ({Horn})", use_container_width=True, help="Simulation-based", type="primary"):
                    st.session_state.factor_nb = Horn
                    perform_FA()

            elif strategy_name == 'FAMD':
                with st.spinner("Analyzing mixed data structure..."):
                    famd_suggested = get_famd_metrics(data_to_analyse)
                    kaiser_famd_number = get_kaiser_famd(data_to_analyse)

                m_col1, m_col2 = st.columns(2)
                m_col1.metric("Suggested Factors", f"{famd_suggested} factors", help="The number of factors required to reach the 70% Cumulative Inertia threshold.")
                m_col2.metric("Kaiser Criterion", f"{kaiser_famd_number} factors", help="A standard approach based on eigenvalues > 1.")

                # Action Buttons
                col1, col2 = st.columns(2)
                
                if col1.button(f"Use 70% Variance ({famd_suggested})", use_container_width=True, type="primary"):
                    st.session_state.factor_nb = famd_suggested
                    perform_FA()

                if col2.button(f"Use Kaiser Criterion ({kaiser_famd_number})", use_container_width=True, type="primary"):
                    st.session_state.factor_nb = kaiser_famd_number
                    perform_FA()

        else:
            factor_nb = st.slider(
                "Select the number of components",
                min_value=1,
                max_value=default_max_components,
                value=default_factor_nb,
                step=1,
                key="factor_nb",
                on_change=perform_FA,
            )

        # Results Section
        if "factor_nb" in st.session_state:
            
            # Debug mode: show all technical details
            if st.session_state.get("show_gpt_calls", True):
                display_results(st)

                expander_FA = st.expander("Factor Analysis results")
                expander_FA.write(st.session_state.df_FA)

                expander_exp = st.expander("Factors components")
                expander_exp.write(pd.DataFrame(st.session_state.components, columns=st.session_state.features_FA,
                        index=[f"Factor {i+1}" for i in range(st.session_state.factor_nb)]))
            
            # Display Q&A if available (shown in both debug and non-debug mode)
            if "QandA" in st.session_state and st.session_state.QandA is not None:
                QandA = st.session_state.QandA
                if isinstance(QandA, pd.DataFrame) and {"User", "Assistant"}.issubset(QandA.columns):
                    for i, row in QandA.iterrows():
                        st.markdown(f"**{row['User']}**")
                        st.markdown(row['Assistant'])
                        st.write("\n")


# Clustering
with tab3:
    if not st.session_state.get("tab2_done", False):
        st.warning("You must complete the factor analysis first!")

    else:
        # Configuration section
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Ask user if they want to automatically find optimal k
            use_elbow = st.radio(
                "Select cluster determination method:",
                options=("Automatic (Elbow Method)", "Manual selection"),
                index=0,
                help=(
                    "The Elbow Method calculates the 'Within-Cluster Sum of Squares' (WCSS). "
                    "It helps you identify the point where adding more clusters no longer "
                    "significantly improves the model, ensuring your groups are statistically distinct."
                )
            )

        with col2:
            if use_elbow == "Automatic (Elbow Method)":
                optimal_k = app_utilities.find_optimal_k_elbow(st.session_state.df_FA)
                st.metric("Optimal Clusters", f"{optimal_k}", help="Calculated using the Elbow Method")
                st.session_state.num_clusters = optimal_k
            else:
                # Slider for number of clusters
                num_clusters = st.slider(
                    "Select the number of clusters",
                    min_value=2,
                    max_value=10,
                    value=default_num_clusters,
                    step=1,
                    key="num_clusters",
                    on_change=perform_clustering,
                )

        # Button to trigger clustering
        if st.button("Run Clustering", use_container_width=True, type="primary"):
            with st.spinner("Analyzing patterns and forming clusters..."):
                perform_clustering()

        # Results section
        factors = [v["label"] for k, v in st.session_state.FA_component_dict.items()]

        if len(factors) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                dimension_x = st.selectbox(
                    "X-axis",
                    factors,
                    key="dim_x",
                    on_change=update_fig_cluster,
                    label_visibility="collapsed"
                )
                st.session_state.dimension_x = dimension_x

            with col2:
                available_for_y = [f for f in factors if f != dimension_x]
                dimension_y = st.selectbox(
                    "Y-axis",
                    available_for_y,
                    key="dim_y",
                    on_change=update_fig_cluster,
                    label_visibility="collapsed"
                )
                st.session_state.dimension_y = dimension_y

            # Create cluster visualization (always 2D)
            vis_cluster = ClusterVisualisation(
                st.session_state.df_FA,
                {k: v["label"] for k, v in st.session_state.FA_component_dict.items()},
                st.session_state.u_labels,
                st.session_state.centroids,
                st.session_state.ind_col_map,
            )
            st.session_state.fig_cluster = vis_cluster.fig
            fig_cluster = st.session_state.get("fig_cluster")
            if fig_cluster is not None and fig_cluster.data:
                st.plotly_chart(fig_cluster, use_container_width=True, theme="streamlit")

            st.session_state.tab3_done = True

        # Cluster description section
        list_cluster_name = st.session_state.get("list_cluster_name")
        list_color_cluster = st.session_state.get("ind_col_map")
        list_description_cluster = st.session_state.get("list_description_cluster")

        if list_color_cluster and list_cluster_name and list_description_cluster:
            for i in list_color_cluster:
                display_cluster_color(list_cluster_name[i], list_color_cluster[i])
                st.write(list_description_cluster[i])


# View
with tab4:
    if not st.session_state.get("tab3_done", False):
        st.warning("You must complete the clustering first!")
        
    else:
        # Get entity data from session state
        label_to_value = st.session_state.get("label_to_value", {})
        option_labels = list(label_to_value.keys()) if label_to_value else []
        
        # Entity selector - defaults to first entity
        if option_labels:
            entity = st.selectbox(
                label="Select entity",
                options=option_labels,
                key="selected_entity",
                index=0,
                on_change=add_to_fig(label_to_value),
                label_visibility="collapsed"
            )

        # Visualization section
        st.plotly_chart(st.session_state.fig_base, use_container_width=True, theme="streamlit")

        if st.session_state.selected_entity == None:
            indice = 0
        else:
            indice = label_to_value[st.session_state.selected_entity]
            
        st.session_state['indice'] = indice

        # Generate entity description
        if 'entity_description' not in st.session_state or st.session_state.get('last_described_entity') != indice:
            wordalisation = CreateWordalisation()
            summary = wordalisation.stream_gpt(prompt_description="Generating entity description")
            st.session_state.entity_description = summary
            st.session_state.last_described_entity = indice
        
        # Display entity description
        st.write(st.session_state.entity_description)

        st.session_state.tab4_done = True
       
     



# debug
# print()

# print("foo")
# for key, value in st.session_state.items():
#     print(key)  # , value)

# print("\t run through")
# for key, value in st.session_state.items():
#     print("\t" + key)  # , value)
