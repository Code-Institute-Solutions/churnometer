
import streamlit as st
import pandas as pd
from src.data_management import load_telco_data, load_pkl_file
from src.machine_learning.predictive_analysis_ui import (
														predict_churn, 
														predict_tenure, 
														predict_cluster)

def page_prospect_body():
	
	# load churn pipleline files
	version = 'v1'
	churn_pipe_dc_fe = load_pkl_file(f'outputs/ml_pipeline/predict_churn/{version}/clf_pipeline_data_cleaning_feat_eng.pkl')
	churn_pipe_model = load_pkl_file(f"outputs/ml_pipeline/predict_churn/{version}/clf_pipeline_model.pkl")
	churn_features = (pd.read_csv(f"outputs/ml_pipeline/predict_churn/{version}/X_train.csv")
					.columns
					.to_list()
					)

	# load tenure pipeline files
	version = 'v1'
	tenure_pipe = load_pkl_file(f"outputs/ml_pipeline/predict_tenure/{version}/clf_pipeline.pkl")
	tenure_labels_map = load_pkl_file(f"outputs/ml_pipeline/predict_tenure/{version}/labels_map.pkl")
	tenure_features = (pd.read_csv(f"outputs/ml_pipeline/predict_tenure/{version}/X_train.csv")
					.columns
					.to_list()
					)
	
	# load cluster pipeline files
	version = 'v1'
	cluster_pipe = load_pkl_file(f"outputs/ml_pipeline/cluster_analysis/{version}/cluster_pipeline.pkl")
	cluster_features = (pd.read_csv(f"outputs/ml_pipeline/cluster_analysis/{version}/TrainSet.csv")
						.columns
						.to_list()
						)
	cluster_profile = pd.read_csv(f"outputs/ml_pipeline/cluster_analysis/{version}/clusters_profile.csv")



	st.write("### Prospect Churnometer Interface")
	st.info(
        f"* The client is interested in determining whether or not a given prospect will churn. "
        f"If so, the client is interested to know when. In addition the client is "
        f"interested in learning from which cluster this prospect will belong in the customer base. "
        f"Based on that, present potential factors that could maintain and/or bring  "
        f"the prospect to a non-churnable cluster."
	)
	st.write(
		f"* Please insert prospect information for predictive analysis: "
		f"Take a look at the main features in the ML pipelines to make sense of"
		f"which feature impacts most which ML pipeline.")

	
	# Generate Live Data
	# check_variables_for_UI(tenure_features, churn_features, cluster_features)
	X_live = DrawInputsWidgets()


	# predict on live data
	st.write("---")
	if st.button("Run Predictive Analysis"): 
		churn_prediction = predict_churn(X_live, churn_features,
										churn_pipe_dc_fe, churn_pipe_model)
		
		if churn_prediction == 1:
			predict_tenure(X_live, tenure_features, tenure_pipe, tenure_labels_map)

		predict_cluster(X_live, cluster_features, cluster_pipe, cluster_profile)
			



def check_variables_for_UI(tenure_features, churn_features, cluster_features):
	import itertools

	# The widgets inputs are the features used in all pipelines (tenure, churn, cluster)
	# We combine them only with unique values
	combined_features = set(
		list(
			itertools.chain(tenure_features, churn_features, cluster_features)
			)
		)
	st.write(f"* There are {len(combined_features)} features for the UI: \n\n {combined_features}")



def DrawInputsWidgets():

	# load dataset
	df = load_telco_data()

    # we create input widgets only for 8 features	
	col1, col2, col3, col4 = st.beta_columns(4)
	col5, col6, col7, col8 = st.beta_columns(4)
	percentageMin, percentageMax = 0.4, 2.0


	# create empy DataFrame, which will be the live data
	X_live = pd.DataFrame([], index=[0]) 


	with col1:
		feature = "Contract"
		st_widget = st.selectbox(
			label= feature,
			options= df[feature].unique()
			)
	X_live[feature] = st_widget


	with col2:
		feature = "InternetService"
		st_widget = st.selectbox(
			label= feature,
			options= df[feature].unique()
			)
	X_live[feature] = st_widget

	with col3:
		feature = "MonthlyCharges"
		st_widget = st.number_input(
			label= feature,
			min_value= df[feature].min()*percentageMin,
			max_value= df[feature].max()*percentageMax,
			value= df[feature].median()
			)
	X_live[feature] = st_widget

	with col4:
		feature = "PaymentMethod"
		st_widget = st.selectbox(
			label= feature,
			options= df[feature].unique()
			)
	X_live[feature] = st_widget


	with col5:
		feature = "OnlineSecurity"
		st_widget = st.selectbox(
			label= feature,
			options= df[feature].unique()
			)
	X_live[feature] = st_widget

	with col6:
		feature = "DeviceProtection"
		st_widget = st.selectbox(
			label= feature,
			options= df[feature].unique()
			)
	X_live[feature] = st_widget




	return X_live
