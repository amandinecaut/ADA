from visualisation_utilities import (
    Visualisation,
    ClusterVisualisation,
    ClusterVisualisation3D,
    DistributionPlot,
)
from chat import EntityChat
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


st.set_page_config(layout="wide")


default_factor_nb, default_threshold, default_max_components, default_num_clusters = (
    get_defaults()
)

height = 1500  # height of the container

for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value
    


# Add and app header
st.title("ADA pipeline")

tab1, tab2, tab3, tab4 = st.tabs(["Load data", "FactorAnalysis", "Clustering", "View"])

FA_done = False

# Load Data
with tab1:

    left_t1, right_t1 = st.columns([0.3, 0.7])
    # Left pane title

    left_t1 = left_t1.container(height=height, border=0)
    right_t1 = right_t1.container(height=height, border=3)

    # clear data button
    #if left_t1.button("Clear data"):
    #    app_utilities.clear_session_state(skip=["file", "map"])

    # run default data
    #data_options = ["Select a Dataset", "Big Five", "World Value Survey", "Football Players", "Breast Cancer","16 Personality", "Cardiovascular Disease"]
    # display the info of the data
    right_t1.markdown("### Data information")

    if "df_full" not in st.session_state:
        right_t1.markdown("Welcome to ADA: the Automatic Data Analyst pipeline")
        right_t1.markdown(":sparkles: Load data to view information :sparkles:")

    with left_t1:
        st.markdown("## 📂 Step 1:  Load Data")
        st.markdown("---")

        load_mode = st.radio(
            "How would you like to load data?",
            options=["🎲 Try a demo dataset", "⬆️ Upload your own"],
            index=0,
            key="load_mode"
        )

        st.markdown("---")

        if load_mode == "🎲 Try a demo dataset":
            st.markdown("#### Choose a demo dataset")
            data_options = ["Select a Dataset"] + list(DATA_PATHS.keys())
            st.selectbox(
                "Available datasets:",
                options=data_options,
                key="demo_dataset_choice",
                index=0,
                on_change=set_default_data_callback
            )

        else:
            st.markdown("#### Upload your dataset")
            st.file_uploader(
                "Upload a CSV file",
                type=["csv"],
                key="file",
                on_change=load_new_data,
            )

            st.markdown("#### Upload column name mapping *(optional)*")
            st.file_uploader(
                "Upload a JSON or Excel mapping file",
                type=["json", "xlsx", "xls"],
                key="map",
                on_change=load_map,
            )
            right_t1.markdown(
            "⚠️ The uploaded dataset must be a **numerical DataFrame** with only numeric columns."
            )
            # Show example dataframe image 
            right_t1.image("./data/example_dataframe.png",caption="Example of a valid numerical DataFrame", width=450)
            right_t1.markdown("The uploaded column mapping could be or a `.json` file, an Excel file (`.xlsx` or `.xls`)")
            # Show example column mapping image 
            right_t1.image("./data/example_json.png", caption="Example of a valid json file", width=450)
            right_t1.markdown("The Excel file (`.xlsx` or `.xls`) must have:\n" "- A first column named **`Key`**\n" "- A second column named **`Value`**")
            right_t1.image("./data/example_xlsx.png", caption="Example of a valid xlsx file", width=450)


        st.markdown("---")
        st.markdown("#### Variable name")
        st.text_input("Enter your variable name:", key="entity_id")
        st.session_state.article = choose_article(st.session_state.entity_id)
  

        if st.session_state.get("data_loading", False):
            st.info("Loading data, please wait...")
        else:
            expander_sample = right_t1.expander("Sample of the data")
            expander_sample.write(st.session_state.df_full.sample(5))

            cols = ["Index"] + st.session_state.df_full.columns.to_list()
            # drop down "select entity", default to "Index"
            entity = right_t1.selectbox(
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
            ignore_cols = right_t1.multiselect(
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
                right_t1.warning(
                    "There are rows containing NaN, these will be dropped."
                )

            expander_nan = right_t1.expander("Rows containing NaN")
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
                right_t1.warning(
                    "Data is empty. Select a different column to index data."
                )
            
            # show st.session_state.df_filtered with the index column blue
            expander_data = right_t1.expander("Show all data to be used")
            update_df(st.session_state.ignore_cols)
            expander_data.write(st.session_state.df_filtered)

            expander_map = right_t1.expander("Column mapping")
            expander_map.write(st.session_state.col_mapping)


            #print("ignore cols", st.session_state.ignore_cols)
            if "entity_radio" not in st.session_state:
                st.session_state.entity_radio = "No"
                
            entity_name_radio = left_t1.radio(
                f"Does your dataset contain a column for the names?",
                options=["Yes", "No"],
                #index=1,  # default to "No"
                key="entity_radio",
                )

            if entity_name_radio == "No":
                st.session_state["col_name"] = None
             
                
            elif entity_name_radio == "Yes":
                if st.session_state.ignore_cols:
                    selected_from_ignore = left_t1.selectbox(
                         f"Pick the column to use for the {st.session_state.entity_id} names",
                         options = st.session_state.ignore_cols,
                         key="selected_from_ignore",
                    )
                   

                     # store it in session_state
                    st.session_state["col_name"] = selected_from_ignore
                    expander_col_name = right_t1.expander(f"Show the column used for the {st.session_state.entity_id} names")
                    #expander_col_name.write(st.session_state.df_full[[selected_from_ignore]])
                    expander_col_name.write(st.session_state.df_full.loc[st.session_state.df_filtered.index, [selected_from_ignore]])
                else:
                     # Handle the case where the user selected "Yes" but there are no ignored columns
                    right_t1.warning("To select an entity name column, it must be in the 'Ignore columns' list. Please select a column to ignore first.")

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
    left_t2, right_t2 = st.columns([0.3, 0.7])
    left_t2 = left_t2.container(height=height, border=0)
    right_t2 = right_t2.container(height=height, border=3)
                
    if not st.session_state.get("tab1_done", False):
        st.warning("You must load your data first!")
    if "df_full" not in st.session_state:
        st.markdown("Load data to perform Factor Analysis")
    elif len(st.session_state.df_filtered) < 10:
        st.markdown("Not enough data to perform Factor Analysis")
    
    else:
        if "analysis" not in st.session_state:
            st.session_state.analysis = None

        left_t2.markdown("## 🧠 Step 2: Factor Analysis")

    # with left_t2:
    #     col1, col2 = st.columns(2)

    #     with col1:
    #         st.markdown("#### 🧩 Factor Analysis")
    #         st.write("Uncover the hidden patterns and latent structures in your data.")
    #         if st.button("Explore Patterns", use_container_width=True):
    #             st.session_state.analysis = "FA"
    #     with col2:
    #         st.markdown("#### 📉 Logistic Regression")
    #         st.write("Predict outcomes and understand the probability of events.")
    #         if st.button("Predict Outcomes", use_container_width=True):
    #             st.session_state.analysis = "LR"
    #     left_t2.markdown("---")

    # Factor analysis choice
    data_to_analyse = st.session_state.df_filtered.loc[:, st.session_state.features]
    strategy_name = select_strategy(data_to_analyse).name
    st.session_state.strategy_name = strategy_name
        
    


    # Number of factors
    left_t2.markdown("##### ✨ **Finding the right balance is the key to a clean model:**")
    left_t2.markdown(get_strategy_description(strategy_name))

    # Ask user if they want to automatically find optimal number of factors
    factor_auto = left_t2.radio(
        "🚀 Smart Suggestion: Should we calculate the ideal factors for you?",
            ("Yes, let's optimize!", "No, I prefer manual control"),
            index=0,
            help="Automatic optimisation uses statistical tests to find the best fit for your data."
        )

    if factor_auto == "Yes, let's optimize!":
           
        if strategy_name in "FA":
            with st.spinner("Calculating optimal factors..."):
                kaiser_number = get_kaiser_criterion()
                Horn = HornParallelAnalysis()
              
            # Using metrics for a more "Dashboard" feel
            m_col1, m_col2 = left_t2.columns(2)
            m_col1.metric("Kaiser Criterion", f"{kaiser_number} Factors", help="A standard approach based on eigenvalues > 1.")
            m_col2.metric("Parallel Analysis", f"{Horn} Factors", help="A more robust statistical simulation method.")

            # Action Buttons
            col1, col2 = left_t2.columns(2)
            if col1.button(f"🎯 Use Kaiser ({kaiser_number})", use_container_width=True, help="Eigenvalues > 1"):
                st.session_state.factor_nb = kaiser_number
                st.toast(f"Applying Kaiser", icon="✅")
                perform_FA()

            if col2.button(f"🚀 Use Parallel Analysis ({Horn})", use_container_width=True, help="Simulation-based"):
                st.session_state.factor_nb = Horn
                st.toast(f"Applying Parallel Analysis ({Horn})", icon="✅")
                perform_FA()


        elif strategy_name == 'FAMD':
            with st.spinner("Analyzing mixed data structure..."):
                famd_suggested = get_famd_metrics(data_to_analyse)
                kaiser_famd_number = get_kaiser_famd(data_to_analyse)

            m_col1, m_col2 = left_t2.columns(2)
            m_col1.metric("Suggested Factors", f"{famd_suggested} factors", help="The number of factors required to reach the 70% Cumulative Inertia threshold.")
            m_col2.metric("Kaiser Criterion", f"{kaiser_famd_number} factors", help="A standard approach based on eigenvalues > 1.")

            # Action Buttons
            col1, col2 = left_t2.columns(2)
                
            if col1.button(f"🚀 Use 70% Variance ({famd_suggested})", use_container_width=True):
                st.session_state.factor_nb = famd_suggested
                st.toast(f"Applying 70% threshold: {famd_suggested} factors", icon="✅")
                perform_FA()

            if col2.button(f"🎯 Use Kaiser ({kaiser_famd_number})", use_container_width=True):
                st.session_state.factor_nb = kaiser_famd_number
                st.toast(f"Applying Kaiser: {kaiser_famd_number} factors", icon="✅")
                perform_FA()

    else:

        factor_nb = left_t2.slider(
                    "Select the number of components",
                    min_value=1,
                    max_value=default_max_components,
                    value=default_factor_nb,
                    step=1,
                    key="factor_nb",
                    on_change=perform_FA,
                )

        
        
    if "factor_nb" in st.session_state:
        left_t2.write(f"Number of components: {st.session_state.factor_nb}.")
        right_t2.write("## Automated labelling")
        display_results(right_t2)

        right_t2.markdown("---")
        right_t2.write("## Factor Analysis results")

        expander_FA = right_t2.expander("Factor Analysis results")
        expander_FA.write(st.session_state.df_FA)
        st.session_state.FA_done = True

        expander_exp = right_t2.expander("Factors components")
            
        expander_exp.write(pd.DataFrame(st.session_state.components,columns=st.session_state.features_FA,
                    index=[f"Factor {i+1}" for i in range(st.session_state.factor_nb)],))

    if st.session_state.get("FA_done", True):
        right_t2.markdown("---")
        right_t2.write("## Question and Answer pairs")
        
        
                
        QandA = create_QandA(text=None)

        if isinstance(QandA, pd.DataFrame) and {"User", "Assistant"}.issubset(QandA.columns):
            for i, row in QandA.iterrows():
                right_t2.markdown(f"### **Question {i+1}:** {row['User']}")
                right_t2.markdown(f"**Answer:** {row['Assistant']}")
                right_t2.write("\n")

        # Path to save the CSV
        QandA_path = "./data/describe/generate/QandA_data.csv"
        # Ensure the folder exists
        os.makedirs(os.path.dirname(QandA_path), exist_ok=True)
        # Save the DataFrame as a CSV file
        QandA.to_csv(QandA_path, index=False)

          

        embeddings = Embeddings()

        directory = os.makedirs(os.path.dirname("./data/embeddings/"), exist_ok=True)
        embed(QandA_path,embeddings)


        st.session_state.tab2_done = True
        
    else:
        st.error("You must complete the Factor Analysis first!")
        
    # left_t2.markdown("---")

    # if st.session_state.analysis == "LR":
    #     right_t2.markdown("---")
    #     left_t2.markdown("### 📉 Logistic Regression")
    #     left_t2.write("#### Work in progress")
    #     right_t2.write("### Work in progress")



# Clustering
with tab3:
    if not st.session_state.get("tab2_done", False):
        st.warning("You must complete the factor analysis first!")
        pass

    else:
        # Create left and right containers
        left_t3, right_t3 = st.columns([0.3, 0.7])
        left_t3 = left_t3.container(height=height, border=0)
        right_t3 = right_t3.container(height=height, border=3)

        left_t3.markdown("## Step 3: Clustering")

        # Ask user if they want to automatically find optimal k
        use_elbow = left_t3.radio(
            "🎯 **Let's find the 'Sweet Spot' for your clusters?**",
            options=("Yes, run the Elbow Method", "No, I'll choose manually"),
            index=0,
            help=(
                "The Elbow Method calculates the 'Within-Cluster Sum of Squares' (WCSS). "
                "It helps you identify the point where adding more clusters no longer "
                "significantly improves the model, ensuring your groups are statistically distinct."
            )
        )   

        if use_elbow == "Yes, run the Elbow Method":
            optimal_k = app_utilities.find_optimal_k_elbow(st.session_state.df_FA)
            left_t3.write(f"The optimal number of clusters is {optimal_k}")
            st.session_state.num_clusters = optimal_k
        
        else:
            # Slider for number of clusters
            num_clusters = left_t3.slider(
                "Select the number of clusters",
                min_value=2,
                max_value=10,
                value=default_num_clusters,
                step=1,
                key="num_clusters",
                on_change=perform_clustering,
            )

        # Button to trigger clustering
        # if left_t3.button("Run Clustering"):
        #     perform_clustering()
        #     right_t3.write("Clustering complete")
        
        if left_t3.button("🚀 Group My Data", use_container_width=True, type="primary"):
            # The spinner will appear inside the left column
            with st.spinner("🔮 Analyzing patterns and forming clusters..."):
                perform_clustering()


            # Success feedback with a "Toast" or an Icon
            st.toast("Clustering complete!", icon="✅")
            right_t3.success("✨ **Clustering Complete!** Your data has been organized into distinct groups.")

            #st.balloons()

        # Factor selection for dimensions
        left_t3.markdown("#### Selection of factors for each dimension")
        factors = [v["label"] for k, v in st.session_state.FA_component_dict.items()]

        if len(factors) < 2:
            right_t3.write(
                "Perform Factor Analysis with at least 2 components to view clustering results"
            )
        elif len(factors) == 2:
            left_t3.write("Only 2 factors available, showing 2D plot")

            dimension_x = left_t3.selectbox(
                "Select a factor for X-axis:",
                factors,
                key="dim_x",
                on_change=update_fig_cluster,
            )
            st.session_state.dimension_x = dimension_x


            dimension_y = left_t3.selectbox(
                "Select a factor for Y-axis:",
                [f for f in factors if f != dimension_x],
                key="dim_y",
                on_change=update_fig_cluster,
            )
            st.session_state.dimension_y = dimension_y

            left_t3.write(f"You selected **{dimension_x}** for X-axis and **{dimension_y}** for Y-axis.")

            # Create cluster visualization
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
                right_t3.plotly_chart(fig_cluster, use_container_width=True, theme="streamlit")

            st.session_state.tab3_done = True 
        else:
            
            plot_type = ["2D", "3D"]
            plot_choice = left_t3.radio("Select plot type", plot_type, key="plot_choice")
           
            # Common dimension selection
            dimension_x = left_t3.selectbox(
                "Select a factor for X-axis:",
                factors,
                key="dim_x",
                on_change=update_fig_cluster if plot_choice == "2D" else update_fig_cluster3d,
            )
            st.session_state.dimension_x = dimension_x

            available_for_y = [f for f in factors if f != dimension_x]
            dimension_y = left_t3.selectbox(
                "Select a factor for Y-axis:",
                available_for_y,
                key="dim_y",
                on_change=update_fig_cluster if plot_choice == "2D" else update_fig_cluster3d,
            )
            st.session_state.dimension_y = dimension_y

            left_t3.write(f"You selected **{dimension_x}** for X-axis and **{dimension_y}** for Y-axis.")

            # 3D specific selection
            if plot_choice == "3D":
                dimension_z = left_t3.selectbox(
                    "Select a factor for Z-axis:",
                    [f for f in factors if f not in [dimension_x, dimension_y]],
                    key="dim_z",
                    on_change=update_fig_cluster3d,
                )
                st.session_state.dimension_z = dimension_z
                left_t3.write(
                    f"You selected **{dimension_x}** for X-axis, **{dimension_y}** for Y-axis, and **{dimension_z}** for Z-axis."
                )

            # Create cluster visualization
            if plot_choice == "2D":
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
                    right_t3.plotly_chart(fig_cluster, use_container_width=True, theme="streamlit")
            else:  # 3D
                vis_cluster3d = ClusterVisualisation3D(
                    st.session_state.df_FA,
                    {k: v["label"] for k, v in st.session_state.FA_component_dict.items()},
                    st.session_state.u_labels,
                    st.session_state.centroids,
                    st.session_state.ind_col_map,
                )
                st.session_state.fig_cluster3d = vis_cluster3d.fig
                fig_cluster3d = st.session_state.get("fig_cluster3d")
                if fig_cluster3d is not None and fig_cluster3d.data:
                    right_t3.plotly_chart(fig_cluster3d, use_container_width=True, theme="streamlit")

            st.session_state.tab3_done = True

        # Cluster description section
        with right_t3:
            st.markdown("<h3><b>Description of each cluster</b></h3>", unsafe_allow_html=True)
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
        left_t4, right_t4 = st.columns([0.3, 0.7])
        left_t4.markdown("## Step 4: View and Summary")

        left_t4.markdown("#### Entity selection")
        # drop down with entity column, default to first column
        entity = left_t4.selectbox(
            label="Entity",
            options=option_labels,
            key="selected_entity",
            index=None,
            placeholder="Select an entity to view details",
            on_change=add_to_fig(label_to_value),
            label_visibility="collapsed"
        )


        with right_t4:
            st.markdown("# Visualisation") 
            st.plotly_chart(st.session_state.fig_base, use_container_width=True, theme="streamlit")

            if st.session_state.selected_entity == None:
                indice = 0
            else:
                indice = label_to_value[st.session_state.selected_entity]
                
            st.session_state['indice'] = indice



            st.markdown("# Summary")   

            # Chat state hash determines whether or not we should load a new chat or continue an old one
            # We can add or remove variables to this hash to change conditions for loading a new chat
            to_hash = (indice)
            # Now create the chat object
            chat = create_chat(to_hash, EntityChat)

    
            
            if chat.state == "empty":

                chat.add_message(
                    "Please can you summarise the data for me?",
                    role="user",
                    user_only=False,
                    visible=False,
                )
                wordalisation = CreateWordalisation()
                summary = wordalisation.stream_gpt()
                st.session_state.entity_description = summary
                chat.add_message(summary)
                chat.state = "default"
            chat.get_input()
            chat.display_messages()
            chat.save_state()

        st.session_state.tab4_done = True
       
     



# debug
# print()

# print("foo")
# for key, value in st.session_state.items():
#     print(key)  # , value)

# print("\t run through")
# for key, value in st.session_state.items():
#     print("\t" + key)  # , value)
