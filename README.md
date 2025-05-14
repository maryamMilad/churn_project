# Telecom Customer Churn Prediction

## Project Overview
This project focuses on predicting customer churn for a telecommunications company using machine learning techniques. Customer churn (also known as customer attrition) refers to when customers stop doing business with a company. Predicting churn helps businesses take proactive measures to retain customers.

The project includes:
- Exploratory data analysis of customer data
- Data preprocessing and feature engineering
- Implementation of multiple machine learning models
- Performance evaluation and comparison of models
- Visualization of results

## Dataset
The dataset contains information about:
- Customer account details (length, area code, state)
- Service usage (day, evening, night, international minutes/calls)
- Service plans (international plan, voicemail plan)
- Customer service calls
- Churn status (target variable)

Training set: 2,666 records  
Test set: 667 records

## Key Features
1. **Data Preprocessing**:
   - Handling categorical variables
   - Feature scaling
   - Correlation analysis
   - PCA for dimensionality reduction

2. **Machine Learning Models**:
   - Logistic Regression
   - Decision Trees
   - Random Forest
   - K-Nearest Neighbors
   - Naive Bayes
   - Support Vector Machines
   - AdaBoost
   - Gradient Boosting

3. **Evaluation Metrics**:
   - Accuracy
   - Precision, Recall, F1-score
   - ROC-AUC
   - Confusion matrices
   - Feature importance

## Results
The Gradient Boosting classifier performed best with:
- Accuracy: 93.1%
- AUC: 0.825
- Precision (churn class): 94%
- Recall (churn class): 66%

## How to Use
1. Clone the repository
2. Install required packages: `pip install -r requirements.txt`
3. Run the Jupyter notebook: `jupyter notebook telecom_churn_prediction.ipynb`

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- plotly
- yellowbrick

## Future Work
- Hyperparameter tuning for better performance
- Implement neural networks
- Deploy as a web service
- Create customer retention strategies based on model insights

## Author
[Your Name]  
[Your Contact Information]  
[Your GitHub Profile]

## License
This project is licensed under the MIT License - see the LICENSE file for details.
