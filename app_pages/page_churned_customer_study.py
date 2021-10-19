import streamlit as st
from src.data_management import load_telco_data

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

from feature_engine.discretisation import ArbitraryDiscretiser
import numpy as np
import plotly.express as px

def page_churned_customer_study_body():

    # load data
    df = load_telco_data()

    # hard copied from data visualization notebook
    vars_to_study = ['Contract', 'InternetService', 'OnlineSecurity', 'TechSupport', 'tenure']

    st.write("### Churned Customer Study")
    st.info(
        f"* The client is interested in understanding the patterns from the customer base, "
        f"so that the client can learn the most relevant variables correlated "
        f"to a churned customer.")


    # inspect collected data
    if st.checkbox("Inspect Customer Base"):
        inspect_data(df)
    st.write("---")


    # Correlation Study Summary
    st.write(
        f"* A correlation study was conducted in the notebook to better understand how "
        f"the variables are correlated to Churn levels. \n"
        f"The most correlated variable are: **{vars_to_study}**"
    )

    st.info(
        f"The correlation indications and plots below interpretation converge. "
        f"It is indicated that: \n"
        f"* A churned customer typically has a month to month contract \n"
        f"* A churned customer typically has fiber optic. \n"
        f"* A churned customer typically doesn't have tech support. \n"
        f"* A churned customer doesn't have online security. \n"
        f"* A churned customer typically has low tenure levels. \n"
    )



    df_eda = df.filter(vars_to_study + ['Churn'])

    # Individual plots per variable
    if st.checkbox("Churn Levels per Variable"):
        churn_level_per_variable(df_eda)
        
    # Parallel plot
    if st.checkbox("Parallel Plot"):
        st.write(
            f"* Information in yellow indicates the profile from a churned customer")
        parallel_plot_churn(df_eda)





def inspect_data(df):
    st.write(
        f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns, find below the first 10 "
        f"rows and a quick inspection of each variable content.")
    st.write(df.head(10))
    
    for col in df.columns: st.write(f"* **{col}**:\n{df[col].unique()}\n")
    




def churn_level_per_variable(df_eda):
    target_var = 'Churn'
    
    for col in df_eda.drop([target_var], axis=1).columns.to_list():
        if df_eda[col].dtype == 'object':
            plot_categorical(df_eda, col, target_var)
        else:
            plot_numerical(df_eda, col, target_var)


def plot_categorical(df, col, target_var):
    fig, axes = plt.subplots(figsize=(12, 5))
    sns.countplot(data=df, x=col, hue=target_var,order = df[col].value_counts().index)
    plt.xticks(rotation=90) 
    plt.title(f"{col}", fontsize=20,y=1.05)        
    st.pyplot(fig) # st.pyplot() renders image, in notebook is plt.show()

def plot_numerical(df, col, target_var):
    fig, axes = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x=col, hue=target_var, kde=True,element="step") 
    plt.title(f"{col}", fontsize=20,y=1.05)
    st.pyplot(fig) # st.pyplot() renders image, in notebook is plt.show()




def parallel_plot_churn(df_eda):
    tenure_map = [-np.Inf, 6, 12, 18, 24, np.Inf]
    disc = ArbitraryDiscretiser(binning_dict={'tenure': tenure_map})
    df_parallel = disc.fit_transform(df_eda)
    
    n_classes = len(tenure_map) - 1
    classes_ranges = disc.binner_dict_['tenure'][1:-1]
    LabelsMap = {}
    for n in range(0,n_classes):
        if n == 0: LabelsMap[n] = f"<{classes_ranges[0]}"
        elif n == n_classes-1: LabelsMap[n] = f"+{classes_ranges[-1]}"
        else: LabelsMap[n] = f"{classes_ranges[n-1]} to {classes_ranges[n]}"


    df_parallel['tenure'] = df_parallel['tenure'].replace(LabelsMap)
    fig = px.parallel_categories(df_parallel, color="Churn", width=750, height=500)
    st.plotly_chart(fig)  # we use st.plotly_chart() to render, in notebook is fig.show()


