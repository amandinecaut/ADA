from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import plotly

import wordalisation
from wordalisation import ClusterWordalisation, Clusterlabel

class Cluster:
    def __init__(self, df, FA_label_map, num_clusters):
        self.df = df
        self.df_values = df.values
        self.FA_label_map = FA_label_map
        self.num_clusters = num_clusters
        self.desc = ClusterWordalisation() 
        self.clustering()

    def clustering(self):
        """Perform K-Means clustering and store results."""

        
        kmeans = KMeans(n_clusters=st.session_state.num_clusters, init='k-means++', max_iter=100, n_init=50, random_state=42)

        
        labels = kmeans.fit_predict(self.df)
        self.df['Cluster'] = labels
        self.centroids = kmeans.cluster_centers_

        self.desc = ClusterWordalisation() 
       
        # Get the cluster description
        self.list_description_cluster = self.get_description_cluster_list(self.centroids)

         # Get the cluster name        
        self.list_cluster_name = self.name_the_cluster(self.list_description_cluster)

      
        self.u_labels = sorted(self.df['Cluster'].unique())
    
        print('DEBUG','len(self.u_labels):', len(self.u_labels))
        colors = plotly.colors.qualitative.Alphabet[:len(self.u_labels)] #plotly.colors.qualitative.Set1[:len(self.u_labels)]
        self.ind_col_map = dict(zip(self.u_labels, colors))

        st.session_state.u_labels, st.session_state.centroids, st.session_state.ind_col_map = self.u_labels , self.centroids,  self.ind_col_map
        st.session_state.df_FA = self.df
        print('color map', st.session_state.ind_col_map)
        st.session_state.list_cluster_name =  self.list_cluster_name
        st.session_state.list_description_cluster = self.list_description_cluster

    def get_description_cluster_list(self, centroids):
      
        list_description_cluster = []
        for i,center in enumerate(centroids):

            self.desc.tell_it_what_data_to_use(center)
            self.desc.messages = self.desc.setup_messages()
          
            description = self.desc.stream_gpt() 
            list_description_cluster.append(description)

            # Convert to DataFrame and save it
           # df = pd.DataFrame(self.desc.messages)
            #new_row = pd.DataFrame([{'role': 'assistant', 'content': description}])
            #df = pd.concat([df, new_row], ignore_index=True)
            #df.to_excel(f"./stream_history/messages_{i}.xlsx", index=False)

        return list_description_cluster
        
    def name_the_cluster(self, list_description_cluster):
        self.labelisation = Clusterlabel()
        
        list_name_cluster = []

        for cluster in list_description_cluster:
            self.labelisation.existing_labels(list_name_cluster)
            self.labelisation.tell_it_what_data_to_use(cluster)
            self.labelisation.messages = self.labelisation.setup_messages()
            label = self.labelisation.stream_gpt()

            list_name_cluster.append(label.lower())
        return list_name_cluster

    

    # Useless for now
    #def find_closest_points(self, kmeans):
    #    """Find indices of the closest points to each cluster center."""
    #    closest_pt_idx = []
    #    for iclust in range(kmeans.n_clusters):
    #        cluster_pts_indices = np.where(kmeans.labels_ == iclust)[0]
    #        cluster_pts = self.df_values.iloc[cluster_pts_indices]
    #        cluster_cen = kmeans.cluster_centers_[iclust]
            # Efficient distance calculation
    #        distances = cdist(cluster_pts, [cluster_cen])
    #        min_idx = np.argmin(distances)
    #        closest_pt_idx.append(cluster_pts_indices[min_idx])
    #    return closest_pt_idx


