import streamlit as st

def predict_churn(X_live, churn_features, churn_pipeline_dc_fe, churn_pipeline_model):

	# from live data, subset features related to this pipeline
	X_live_churn = X_live.filter(churn_features)
	
	# apply data cleaning / feat engine pipeline to live data
	X_live_churn_dc_fe = churn_pipeline_dc_fe.transform(X_live_churn)
	
	# predict
	churn_prediction = churn_pipeline_model.predict(X_live_churn_dc_fe)
	churn_prediction_proba = churn_pipeline_model.predict_proba(X_live_churn_dc_fe)


	# Create a logic to display the results
	churn_chance = churn_prediction_proba[0,churn_prediction][0]*100
	if churn_prediction == 1: churn_result = 'will'
	else: churn_result = 'will not'

	statement = (
		f'### There is {churn_chance.round(1)}% probability '
		f'that this prospect **{churn_result} churn**.')


	st.write(statement)

	return churn_prediction





def predict_tenure(X_live, tenure_features, tenure_pipeline, tenure_labels_map):

	# from live data, subset features related to this pipeline
	X_live_tenure = X_live.filter(tenure_features)

	# predict
	tenure_prediction = tenure_pipeline.predict(X_live_tenure)
	tenure_prediction_proba = tenure_pipeline.predict_proba(X_live_tenure)

	# create a logic to display the results
	proba = tenure_prediction_proba[0,tenure_prediction][0]*100
	tenure_levels = tenure_labels_map[tenure_prediction[0]]

	statement = (
		f"* In addition, there is a {proba.round(2)}% probability the prospect "
		f"will stay **{tenure_levels} months**. ")

	st.write(statement)



def predict_cluster(X_live, cluster_features, cluster_pipeline, cluster_profile):

	# from live data, subset features related to this pipeline
	X_live_cluster = X_live.filter(cluster_features)

	# predict 
	cluster_prediction = cluster_pipeline.predict(X_live_cluster)



	statement = (
		f"### The prospect is expected to belong to **cluster {cluster_prediction[0]}** \n"
		f" We consider **cluster 1 as churnable** and **cluster 3 as almost churnable**. "
		f" We consider **clusters 0 and 2 as non-churnable** \n"
		f"* Consider the cluster profile below and the existing product offers to "
		f"suggest a plan so that the prospect can move to a better or a non-churnable cluster.")
	st.write("---")
	st.write(statement)


	# hack to not display index in st.table() or st.write()
	cluster_profile.index = [" "] * len(cluster_profile)
	# display cluster profile in a table - it is better than in st.write() 
	st.table(cluster_profile)

