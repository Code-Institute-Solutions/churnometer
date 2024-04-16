## How to use this repo

1. Fork this repo and copy the https URL of your forked churnometer repo

1. Log into the cloud IDE with your GitHub account.

1. On your Dashboard, click on the New Workspace button

1. Paste in the URL you copied from GitHub earlier

1. Click Create

1. Wait for the workspace to open. This can take a few minutes.

1. Open a new terminal and `pip3 install -r requirements.txt`

1. Click the kernel button and choose Python Environments.

1. Choose the kernel Python 3.8.18 as it inherits from the workspace, so it will be Python-3.8.18 as installed by our template. To confirm this, you can use `! python --version` in a notebook code cell.

Your workspace is now ready to use. When you want to return to this project, you can find it in your Cloud IDE Dashboard. You should only create 1 workspace per project.

## Dataset Content

The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/telecom-churn-dataset). We created then a fictitious user story where predictive analytics can be applied in a real project in the workplace.
Each row represents a customer, each column contains a customer attribute. The data set includes information about:
- Services that each customer has signed up for, like phone, multiple lines, internet, online security, online backup, device protection, tech support, streaming TV and movies
- Customer information, like how long they've been a customer if they churned out, their contract type, payment method, paperless billing, monthly charges, and total charges
- Customer profile, like gender, if they have partners and dependents


| Variable         | Meaning                                                     | Units                                                                                |
|------------------|-------------------------------------------------------------|--------------------------------------------------------------------------------------|
| customerID       | Customer identification                                     | Number and Letters code that form a unique identifier for a customer                  |
| gender           | Customer gender                                             | Female or Male                                                                       |
| SeniorCitizen    | Customer is a senior citizen or not                         | 1 for Senior and 0 for not Senior                                                    |
| Partner          | Customer has a partner or not                               | Yes or No                                                                            |
| Dependents       | Customer has dependents or not                              | Yes or No                                                                            |
| tenure           | Number of months the customer has stayed with the company   | 0 to 72                                                                              |
| PhoneService     | Customer has a phone service or not                         | Yes or No                                                                            |
| MultipleLines    | Customer has multiple lines or not                          | Yes, No, No phone service                                                            |
| InternetService  | Customer has an internet service provider                   | DSL, Fiber optic, No                                                                 |
| OnlineSecurity   | Customer has online security or not                         | Yes, No, No internet service                                                         |
| OnlineBackup     | Customer has online backup or not                           | Yes, No, No internet service                                                         |
| DeviceProtection | Customer has device protection or not                       | Yes, No, No internet service                                                         |
| TechSupport      | Customer has tech support or not                            | Yes, No, No internet service                                                         |
| StreamingTV      | Customer has streaming TV or not                            | Yes, No, No internet service                                                         |
| StreamingMovies  | Customer has streaming movies or not                        | Yes, No, No internet service                                                         |
| Contract         | Contract term of the customer                               | Month-to-month, One year, Two year                                                   |
| PaperlessBilling | Customer has paperless billing or not                       | Yes, No                                                                              |
| PaymentMethod    | Customer’s payment methods                                   | Electronic check, Mailed check, Bank transfer (automatic), Credit card   (automatic) |
| MonthlyCharges   | Amount charged to the customer monthly                      | 18.3 - 119                                                                           |
| TotalCharges     | Total amount charged as a customer of our company           | 18.8 - 8.68k                                                                         |
| Churn            | Customer churned or not                                     | Yes or No                                                                            |


## Project Terms & Jargon
	- A customer is a person who consumes your service or product.
	- A prospect is a potential customer.
	- A churned customer is a user who has stopped using your product or service.
	- This customer has a tenure level, which is the number of months this person has used our product/service.

## Business Requirements
As a Data Analyst from Code Institute Consulting, you are requested by the Telco division to provide actionable insights and data-driven recommendations to a Telecom corporation. This client has a substantial customer base and is interested in managing churn levels and understanding how the sales team could better interact with prospects. The client has shared the data.

- 1 - The client is interested in understanding the patterns from the customer base so that the client can learn the most relevant variables correlated to a churned customer.
- 2 - The client is interested in determining whether or not a given prospect will churn. If so, the client is interested to know when. In addition, the client is interested in learning from which cluster this prospect will belong in the customer base. Based on that, present potential factors that could maintain and/or bring the prospect to a non-churnable cluster.


## Hypothesis and how to validate?
- 1 - We suspect customers are churning with low tenure levels.
	- A Correlation study can help in this investigation
- 2 - A customer survey showed our customers appreciate Fibre Optic.
	- A Correlation study can help in this investigation


## The rationale to map the business requirements to the Data Visualizations and ML tasks
- **Business Requirement 1:** Data Visualization and Correlation study
	- We will inspect the data related to the customer base.
	- We will conduct a correlation study (Pearson and Spearman) to understand better how the variables are correlated to Churn.
	- We will plot the main variables against Churn to visualize insights.

- **Business Requirement 2:** Classification, Regression, Cluster and Data Analysis
	- We want to predict if a prospect will churn or not. We want to build a binary classifier.
	- We want to predict the tenure level for a prospect that is expected to churn. We want to build a regression model or change the ML task to classification depending on the regressor performance.
	- We want to cluster similar customers to predict from which cluster a prospect will belong.
	- We want to understand a cluster profile to present potential options to maintain or bring the prospect to a non-churnable cluster.




## ML Business Case

### Predict Churn
#### Classification Model
- We want an ML model to predict if a prospect will churn based on historical data from the customer base, which doesn't include tenure and total charges since these values are zero for a prospect. The target variable is categorical and contains 2-classes. We consider a **classification model**. It is a supervised model, a 2-class, single-label, classification model output: 0 (no churn), 1 (yes churn)
- Our ideal outcome is to provide our sales team with reliable insight into onboarding customers with a higher sense of loyalty.
- The model success metrics are
	- at least 80% Recall for Churn, on train and test set 
	- The ML model is considered a failure if:
		- after 3 months of usage, more than 30% of newly onboarded customers churn (it is an indication that the offers are not working or the model is not detecting potential churners)
		- Precision for no Churn is lower than 80% on train and test set. (We don't want to offer a free discount to many non-churnable prospects)
- The model output is defined as a flag, indicating if a prospect will churn or not and the associated probability of churning. If the prospect is online, the prospect will have already provided the input data via a form. If the prospect talks to a salesperson, the salesperson will interview to gather the input data and feed it into the App. The prediction is made on the fly (not in batches).
- Heuristics: Currently, there is no approach to predict churn on prospects
- The training data to fit the model comes from the Telco Customer. This dataset contains about 7 thousand customer records.
	- Train data - target: Churn; features: all other variables, but tenure, total charges and customerID

### Predict Tenure
#### Regression Model
- We want an ML model to predict tenure levels, in months, for a prospect expected to churn. A target variable is a discrete number. We consider a **regression model**, which is supervised and uni-dimensional.
- Our ideal outcome is to provide our sales team with reliable insight into onboarding customers with a higher sense of loyalty.
- The model success metrics are
	- At least 0.7 for R2 score, on train and test set
	- The ML model is considered a failure if:
		- after 12 months of usage, the model's predictions are 50% off more than 30% of the time. Say, a prediction is >50% off if predicted 10 months and the actual value was 2 months.
- The output is defined as a continuous value for tenure in months. It is assumed that this model will predict tenure if the Predict Churn Classifier predicts 1 (yes for churn). If the prospect is online, the prospect will have already provided the input data via a form. If the prospect talks to a salesperson, the salesperson will interview to gather the input data and feed it into the App. The prediction is made on the fly (not in batches).
- Heuristics: Currently, there is no approach to predict the tenure levels for a prospect.
- The training data to fit the model comes from the Telco Customer. This dataset contains about 7 thousand customer records.
	- Train data - filter data where Churn == 1, then drop the Churn variable. Target: tenure; features: all other variables, but total charges and customerID


#### Classification Model
- Before the analysis, we visualized a Regressor pipeline to predict Churn; however, the performance didn’t meet the requirement (at least 0.7 for R2 score, on train and test set)
- We used a technique to convert the ML task from Regression to Classification. We discretized the target into 3 ranges: <4 months, 4-20 months and +20 months. 
- The classification pipeline can detect a prospect that would likely churn in less than four months and a prospect that would likely churn in more than 20 months.
- A target variable is categorical and contains 3 classes. We consider a **classification model**, which is supervised and uni-dimensional.
- Our ideal outcome is to provide our sales team with reliable insight into onboarding customers with a higher sense of loyalty.
- The model success metrics are
	- At least 0.8 Recall for <4 months, on train and test set
	- The ML model is considered a failure if:
		- after 3 months of usage, more than 30% of customers that were expected to churn in <4 months do not churn
- The output is defined as a class, which maps to a range of tenure in months. It is assumed that this model will predict tenure if the Predict Churn Classifier predicts 1 (yes for churn). If the prospect is online, the prospect will have already provided the input data via a form. If the prospect talks to a salesperson, the salesperson will interview to gather the input data and feed it into the App. The prediction is made on the fly (not in batches).
- Heuristics: Currently, there is no approach to predict the tenure levels for a prospect.
- The training data to fit the model comes from the Telco Customer. This dataset contains about 7 thousand customer records.
	- Train data - filter data where Churn == 1, then drop the Churn variable. Target: tenure; features: all other variables, but total charges and customerID


### Cluster Analysis
#### Clustering Model
- We want an ML model to cluster similar customer behaviour. It is an unsupervised model.
- Our ideal outcome is to provide our sales team with reliable insight into onboarding customers with a higher sense of loyalty.
- The model success metrics are
	- at least 0.45 for the average silhouette score
	- The ML model is considered a failure if the model suggests from more than 15 clusters (might become too difficult to interpret in practical terms)
- The output is defined as an additional column appended to the dataset. This column represents the cluster's suggestions. It is a categorical and nominal variable represented by numbers starting at 0.
- Heuristics: Currently, there is no approach to grouping similar customers
- The training data to fit the model comes from the Telco Customer. This dataset contains about 7 thousand customer records.
	- Train data - features: all variables, but customerID, TotalCharges, Churn, and tenure 


## Dashboard Design (Streamlit App User Interface)

### Page 1: Quick project summary
- Quick project summary
	- Project Terms & Jargon
	- Describe Project Dataset
	- State Business Requirements

### Page 2: Customer Base Churn Study
- Before the analysis, we knew we wanted this page to answer business requirement 1, but we couldn't know in advance which plots would need to be displayed.
- After data analysis, we agreed with stakeholders that the page will: 
	- State business requirement 1
	- Checkbox: data inspection on customer base (display the number of rows and columns in the data, and display the first ten rows of the data)
	- Display the most correlated variables to churn and the conclusions
	- Checkbox: Individual plots showing the churn levels for each correlated variable 
	- Checkbox: Parallel plot using Churn and correlated variables 

### Page 3: Prospect Churnometer
- State business requirement 2
- Set of widgets inputs, which relates to the prospect profile. Each set of inputs is related to a given ML task to predict prospect Churn, Tenure and Cluster.
- "Run predictive analysis" button that serves the prospect data to our ML pipelines and predicts if the prospect will churn or not, if so, when. It also shows to which cluster the prospect belongs and the cluster's profile. For the churn and tenure predictions, the page will inform the associated probability for churning and tenure level.

### Page 4: Project Hypothesis and Validation
- Before the analysis, we knew we wanted this page to describe each project hypothesis, the conclusions, and how we validated each. After the data analysis, we can report that:
- 1 - We suspect customers are churning with low tenure levels
	- Correct. The correlation study at Churned Customer Study supports that.
- 2 -  A customer survey showed our customers appreciate Fibre Optic.
	- A churned user typically has Fiber Optic, as demonstrated by a Churned Customer Study. The insight will be taken to the survey team for further discussions and investigations.

### Page 5: Predict Churn
- Considerations and conclusions after the pipeline is trained
- Present ML pipeline steps
- Feature importance
- Pipeline performance

### Page 6: Predict Tenure
- Considerations and conclusions after the pipeline is trained
- Present ML pipeline steps
- Feature importance
- Pipeline performance

### Page 7: Cluster Analysis
- Considerations and conclusions after the pipeline is trained
- Present ML pipeline steps
- Silhouette plot
- Clusters distribution across Churn levels
- Relative Percentage (%) of Churn in each cluster
- The most important features to define a cluster
- Cluster Profile

