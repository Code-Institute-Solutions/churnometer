import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from src.data_management import load_telco_data, load_pkl_file


def page_cluster_body():

    # load cluster analysis files and pipeline
    version = 'v1'
    cluster_pipe = load_pkl_file(
        f"outputs/ml_pipeline/cluster_analysis/{version}/cluster_pipeline.pkl")
    cluster_silhouette = plt.imread(
        f"outputs/ml_pipeline/cluster_analysis/{version}/clusters_silhouette.png")
    features_to_cluster = plt.imread(
        f"outputs/ml_pipeline/cluster_analysis/{version}/features_define_cluster.png")
    cluster_profile = pd.read_csv(
        f"outputs/ml_pipeline/cluster_analysis/{version}/clusters_profile.csv")
    cluster_features = (pd.read_csv(f"outputs/ml_pipeline/cluster_analysis/{version}/TrainSet.csv")
                        .columns
                        .to_list()
                        )

    # dataframe for cluster_distribution_per_variable()
    df_churn_vs_clusters = load_telco_data().filter(['Churn'], axis=1)
    df_churn_vs_clusters['Clusters'] = cluster_pipe['model'].labels_

    st.write("### ML Pipeline: Cluster Analysis")
    # display pipeline training summary conclusions
    st.info(
        f"* We refitted the cluster pipeline using fewer variables, and it delivered equivalent "
        f"performance to the pipeline fitted using all variables.\n"
        f"* The pipeline average silhouette score is 0.68"
    )
    st.write("---")

    st.write("#### Cluster ML Pipeline steps")
    st.write(cluster_pipe)

    st.write("#### The features the model was trained with")
    st.write(cluster_features)

    st.write("#### Clusters Silhouette Plot")
    st.image(cluster_silhouette)

    cluster_distribution_per_variable(df=df_churn_vs_clusters, target='Churn')

    st.write("#### Most important features to define a cluster")
    st.image(features_to_cluster)

    # text based on "07 - Modeling and Evaluation - Cluster Sklearn" notebook conclusions
    st.write("#### Cluster Profile")
    statement = (
        f"* Historically, **users in Clusters 0 do not tend to Churn**, "
        f"whereas in **Cluster 1 a third of users churned**, "
        f"and in **Cluster 2 a quarter of users churned**. \n"
        f"* From the Predict Churn study, we noticed that the ContractType and InternetService "
        f"are the predictor variables to determine, if a person will churn or not.\n"
        f"* **One potential action** when you detect that a given prospect is expected to churn and "
        f"will belong to cluster 1 or 2 is to mainly avoid month to month contract type, "
        f"like we learned in the churned customer study. \n"
        f"* The salesperson would have then to consider the current product and services "
        f"plan availability and encourage the prospect to move to another contract."
    )
    st.info(statement)

    # text based on "07 - Modeling and Evaluation - Cluster Sklearn" notebook conclusions
    statement = (
        f"* The cluster profile interpretation allowed us to label the cluster in the following fashion:\n"
        f"* Cluster 0 has users without internet, who are low spenders with a phone.\n"
        f"* Cluster 1 has users with Internet, who are high spenders with a phone.\n"
        f"* Cluster 2 has users with Internet, who are mid spenders without a phone."
    )
    st.success(statement)

    # hack to not display the index in st.table() or st.write()
    cluster_profile.index = [" "] * len(cluster_profile)
    st.table(cluster_profile)


# code coped from "07 - Modeling and Evaluation - Cluster Sklearn" notebook - under "Cluster Analysis" section
def cluster_distribution_per_variable(df, target):

    df_bar_plot = df.value_counts(["Clusters", target]).reset_index()
    df_bar_plot.columns = ['Clusters', target, 'Count']
    df_bar_plot[target] = df_bar_plot[target].astype('object')

    st.write(f"#### Clusters distribution across {target} levels")
    fig = px.bar(df_bar_plot, x='Clusters', y='Count',
                 color=target, width=800, height=350)
    fig.update_layout(xaxis=dict(tickmode='array',
                      tickvals=df['Clusters'].unique()))
    # we replaced fig.show() for a streamlit command to render the plot
    st.plotly_chart(fig)

    df_relative = (df
                   .groupby(["Clusters", target])
                   .size()
                   .groupby(level=0)
                   .apply(lambda x:  100*x / x.sum())
                   .reset_index()
                   .sort_values(by=['Clusters'])
                   )
    df_relative.columns = ['Clusters', target, 'Relative Percentage (%)']

    st.write(f"#### Relative Percentage (%) of {target} in each cluster")
    fig = px.line(df_relative, x='Clusters', y='Relative Percentage (%)',
                  color=target, width=800, height=350)
    fig.update_layout(xaxis=dict(tickmode='array',
                      tickvals=df['Clusters'].unique()))
    fig.update_traces(mode='markers+lines')
    # we replaced fig.show() for a streamlit command to render the plot
    st.plotly_chart(fig)
