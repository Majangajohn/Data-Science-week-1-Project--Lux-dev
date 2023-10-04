Week 1  project: Churn Prediction for Sprint

Imagine you're working with Sprint, one of the biggest telecom companies in the USA. They're really keen on figuring out how many customers might decide to leave them in the coming months. Luckily, they've got a bunch of past data about when customers have left before, as well as info about who these customers are, what they've bought, and other things like that.

 Working with Sprint to predict customer churn is a crucial task for telecom companies like them. To predict customer churn effectively, we can employ a predictive analytics approach using the historical data they have.

Creating a full churn prediction system in Python involves a significant amount of code and can vary based on the specific libraries and tools you choose to use.

Solution:
for this case i will be using Random Forest Classifier model for the churn prediction task which is a popular ensemble learning method that combines the predictions of multiple decision trees to improve predictive accuracy and reduce overfitting, making it a suitable choice for churn prediction tasks.


Certainly, here are the key steps used in the Python code for churn prediction:

1. **Data Loading:**
   - Load historical churn data from a CSV file into a Pandas DataFrame.
    ```python
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    # Load the historical churn data (assuming you have a CSV file)
    data = pd.read_csv('Customer_data.csv')

    ```

2. **Data Preprocessing:**
   - This step typically involves handling missing values, encoding categorical variables, and performing feature engineering. However, the code provided simplifies this part and assumes that preprocessing has been done beforehand.
   ```python
    # handle missing values
    data.dropna(inplace = True)

    # checking on data structure
    data.info()
    #dropping columns which are not of importance 
    data = data.drop(['customer_id','first_name','last_name','email','postal_code'], axis = 1)
    #encoding categorical variable using one-hot coding
    encoded_data = pd.get_dummies(data[['country','purchase_history','Churn']],drop_first = True).astype(int)
    #creating separate dta frame for float and int columns
    int_df = data[['age','monthly_payment','contract_length','data_usage','customer_service_rating']]
    # combining 2 data sets into 1 data frame along columns
    final_df = pd.concat([int_df, encoded_data],axis = 1)  

    # Assuming 'Churn' is the target variable
    X = final_df.drop('Churn', axis=1)
    y = final_df['Churn']

   ```

3. **Data Splitting:**
   - Split the data into training and test sets using scikit-learn's `train_test_split` function. This is essential to evaluate the model's performance on unseen data.
       ```python
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

4. **Model Selection and Training:**
   - Choose a machine learning model for churn prediction. In this code, a Random Forest Classifier is used.
   - Instantiate the model with specified hyperparameters (e.g., number of trees) and train it using the training data.
```python
   # Model selection and training
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
```

5. **Model Evaluation:**
   - Make predictions on the test set using the trained model.
   - Evaluate the model's performance using various classification metrics, including accuracy, precision, recall, F1-score, and ROC AUC score.
   - Print the evaluation results to assess how well the model is performing.
   
```python
   # Model evaluation
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC AUC Score:", roc_auc)
```

6. **Model Deployment (Optional):**
   - Save the trained model to a file using joblib. This allows you to use the model for making predictions on new data in the future.

Please note that this code is a simplified example for educational purposes. In a real-world scenario, additional steps such as hyperparameter tuning, cross-validation, and more extensive data preprocessing would be necessary to build a robust churn prediction system.

```python
# Deployment: Save the trained model for future predictions
import joblib
joblib.dump(clf, 'churn_model.pkl')

```