import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

data=pd.read_csv("Employee.csv")


# data.head()

data.dropna(inplace=True)

data = pd.get_dummies(data)


#print(data.columns)

X = data.drop('Attrition_Yes', axis=1)
y = data['Attrition_Yes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))


# Predict attrition for all employee
attrition_probabilities = rf_model.predict_proba(X)[:, 1]

# Add attrition probabilities to the original DataFrame
data['Attrition_Probability'] = attrition_probabilities

# Sort employees by attrition probability in descending order
employees_most_likely_to_leave = data.sort_values(by='Attrition_Probability', ascending=False)

# Display the top N employees most likely to leave
N = 20  # Number of employees to display
print(employees_most_likely_to_leave.head(N))