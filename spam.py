import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load your dataset (replace 'your_dataset.csv' with your filename)
dataset = pd.read_csv('spam.csv',encoding='latin-1')

# Assign columns
X = dataset['v2']  # Email content
y = dataset['v1']  # Labels ('ham' or 'spam')

# Split data into training and testing sets for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)




# Create a vectorizer to convert text into numerical representations
vectorizer = CountVectorizer()

# Fit and transform the training data
X_train_transformed = vectorizer.fit_transform(X_train)

# Transform the testing data using the same vocabulary
X_test_transformed = vectorizer.transform(X_test)


# Create a Naive Bayes classifier (well-suited for text classification)
model = MultinomialNB()

# Train the model on the training data
model.fit(X_train_transformed, y_train)

def predict_spam(email_text):
    """Predicts if an email is spam or ham.

    Args:
        email_text (str): The content of the email.

    Returns:
        str: 'spam' or 'ham'
    """

    # Preprocess the input email
    email_transformed = vectorizer.transform([email_text])

    # Make a prediction
    prediction = model.predict(email_transformed)[0]
    return prediction 

# Example usage
new_email = input("enter content to check spam or not: ")
prediction = predict_spam(new_email)
if(prediction=="ham"):
    print("Prediction:", "not spam")
else:
    print("Prediction:", prediction)
    

# Make predictions on the test set
y_pred = model.predict(X_test_transformed)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

