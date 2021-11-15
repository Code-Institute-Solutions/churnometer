import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_churned_customer_study import page_churned_customer_study_body
from app_pages.page_prospect import page_prospect_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body
from app_pages.page_predict_churn import page_predict_churn_body
from app_pages.page_predict_tenure import page_predict_tenure_body
from app_pages.page_cluster import page_cluster_body

app = MultiPage(app_name= "Churnometer") # Create an instance of the app 

# Add your app pages here using .add_page()
app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Customer Base Churn Study", page_churned_customer_study_body)
app.add_page("Prospect Churnometer", page_prospect_body)
app.add_page("Project Hypothesis and Validation", page_project_hypothesis_body)
app.add_page("ML: Prospect Churn", page_predict_churn_body)
app.add_page("ML: Prospect Tenure", page_predict_tenure_body)
app.add_page("ML: Cluster Analysis", page_cluster_body)

app.run() # Run the  app
