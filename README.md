## Dataset Content
The dataset is sourced from [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn). We created then a fictitious user story where predictive analytics can be applied in a real project in the workplace.
Each row represents a customer, each column contains a customer attribute. The data set includes information about:
* Services that each customer has signed up for, like phone, multiple lines, internet, online security, online backup, device protection, tech support, streaming TV and movies
* Customer information, like how long they've been a customer if they churned out, their contract type, payment method, paperless billing, monthly charges, and total charges
* Customer profile, like gender, if they have partners and dependents


| Variable         | Meaning                                                     | Units                                                                                |
|------------------|-------------------------------------------------------------|--------------------------------------------------------------------------------------|
| customerID       | Customer identification                                     | Number and Letters code that form a unique identifier to a customer                  |
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
| PaymentMethod    | Customer’s payment method                                   | Electronic check, Mailed check, Bank transfer (automatic), Credit card   (automatic) |
| MonthlyCharges   | Amount charged to the customer monthly                      | 18.3 - 119                                                                           |
| TotalCharges     | Total amount charged as a customer of our company           | 18.8 - 8.68k                                                                         |
| Churn            | Customer churned or not                                     | Yes or No                                                                            |


* **Project Terms & Jargon**
	* A customer is a person who consumes your service or product.
	* A prospect is a potential customer.
	* A churned customer is a user who has stopped using your product or service.
	* This customer, has a tenure level, which is the number of months this person has used our product/service.

## Business Requirements
As a Data Analyst from Code Institute Consulting, you are requested by the Telco division to provide actionable insights and data-driven recommendations to a Telecom corporation. This client has a substantial customer base and is interested in managing churn levels and understanding how the sales team could better interact with prospects. The client has shared the data.

* 1 - The client is interested in understanding the patterns from the customer base so that the client can learn the most relevant variables correlated to a churned customer.
* 2 - The client is interested in determining whether or not a given prospect will churn. If so, the client is interested to know when. In addition, the client is interested in learning from which cluster this prospect will belong in the customer base. Based on that, present potential factors that could maintain and/or bring the prospect to a non-churnable cluster.


## Hypothesis and how to validate?
* We suspect customers are churning with low tenure levels.
	* A Correlation study can help in this investigation
* A customer survey showed our customers appreciate fibre Optic.
	* A Correlation study can help in this investigation


## Rationale to map the business requirements to the Data Visualizations and ML tasks
* **Business Requirement 1**: Data Visualization and Correlation study
	* We will inspect the data related to the customer base.
	* We will conduct a correlation study (Pearson and Spearman) to better understand how the variables are correlated to Churn.
	* We will plot the main variables against Churn to visualize insights.


* **Business Requirement 2**:  Classification, Regression, Cluster, Data Analysis
	* We want to predict if a prospect will churn or not. We want to build a binary classifier.
	* We want to predict tenure level for a prospect that is expected to churn. We want to build a regression model or, depending on the regressor performance, change the ML task to classification.
	* We want to cluster similar customers, to predict from which cluster a prospect will belong.
	* We want to understand a cluster profile to present potential options that could maintain or bring the prospect to a non-churnable cluster.




## ML Business Case

### Predict Churn
#### Classification Model
* We want an ML model to predict if a prospect will churn based on historical data from the customer base, which doesn't include tenure and total charges since these values are zero for a prospect. The target variable is categorical and contains 2-classes. We consider a **classification model**. It is a supervised model, a 2-class, single-label, classification model output: 0 (no churn), 1 (yes churn)
* Our ideal outcome is to provide our sales team with reliable insight into how to onboard customers with a higher sense of loyalty.
* The model success metrics are
	* at least 80% Recall for Churn, on train and test set 
	* The ML model is considered a failure if:
		* after 3 months of usage, more than 30% of newly onboarded customers churn (it is an indication that the offers are not working or the model is not detecting potential churners)
		* Precision for no Churn is lower than 80% on train and test set. (We don't want to offer a free discount to many non-churnable prospects)
* The model output is defined as a flag, indicating if a prospect will churn or not and the associate probability of churning. If the prospect is online, the prospect will have already provided the input data via a form. If the prospect talks to a salesperson, the salesperson will interview to gather the input data and feed it into the App. The prediction is made on the fly (not in batches).
* Heuristics: Currently there is no approach to predict churn on prospects
* The training data to fit the model come from the Telco Customer. This dataset contains about 7 thousand customer records.
	* Train data - target: Churn; features: all other variables, but tenure, total charges and customerID

### Predict Tenure
#### Regression Model
* We want an ML model to predict tenure levels, in months, for a prospect expected to churn. A target variable is a discrete number. We consider a **regression model**, which is supervised and uni-dimensional.
* Our ideal outcome is to provide our sales team with reliable insight into onboarding customers with a higher sense of loyalty.
* The model success metrics are
	* At least 0.7 for R2 score , on train and test set
	* The ML model is considered a failure if:
		* after 12 months of usage, the model's predictions are 50% off more than 30% of the time. Say, a prediction is >50% off if predicted 10 months and the actual value was 2 months.
* The output is defined as a continuous value for tenure in months. It is assumed that this model will predict tenure if the Predict Churn Classifier predicts 1 (yes for churn). If the prospect is online, the prospect will have already provided the input data via a form. If the prospect talks to a salesperson, the salesperson will interview to gather the input data and feed it into the App. The prediction is made on the fly (not in batches).
* Heuristics: Currently there is no approach to predict tenure levels on prospect.
* The training data to fit the model come from the Telco Customer. This dataset contains about 7 thousand customer records.
	* Train data - filter data where Churn == 1, then drop Churn variable. Target: tenure; features: all other variables, but total charges and customerID


### Cluster Analysis
#### Clustering Model
* We want an ML model to cluster similar customer behavior. It is an unsupervised model.
* Our ideal outcome is to provide our sales team with reliable insight into onboarding customers with a higher sense of loyalty.
* The model success metrics are
	* at least 0.45 for silhouette score
	* The ML model is considered a failure if: model suggests from more than 15 clusters (might become too difficult to interpret in practical terms)
* The output is defined as an additional column appended to the dataset. This column represents the clusters suggestions. It is a categorical and nominal variable, represented by numbers, starting at 0.
* Heuristics: Currently there is no approach to group similar customers
* The training data to fit the model come from the Telco Customer. This dataset contains about 7 thousand customer records.
	* Train data - features: all variables, but customerID, TotalCharges, Churn, and tenure 


## Dashboard Design (Streamlit App User Interface)

### Page 1: Quick project summary
* Quick project summary
	* Project Terms & Jargon
	* Describe Project Dataset
	* State Business Requirements

### Page 2: Customer Base Churn Study
* It will answer business requirement 1

### Page 3: Prospect Churnometer
* User Interface with prospect inputs and predictions indicating if the prospect will churn or not, if so, when, to which cluster the prospect belongs, and an indication on which cluster the prospect belong to.
* In addition, present cluster profile; so the person serving the prospect can suggest an offer that will bring the prospect to a non-churnable customer.

### Page 4: Project Hypothesis and Validation
* For each project hypothesis, describe the conclusion and how you validated it.

### Page 5: Predict Churn
* Present ML pipeline steps
* Feature importance
* Pipeline performance

### Page 6: Predict Tenure
* Present ML pipeline steps
* Feature importance
* Pipeline performance

### Page 7: Cluster Analysis
* Present ML pipeline steps
* Silhouette score
* Clusters distribution across Churn levels
* Relative Percentage (%) of Churn in each cluster
* Most important features to define a cluster
* Cluster Profile

