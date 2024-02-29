#pandas- working with dataset
#numpy- working with arrays
#sklearn- models & statistical modelling
#sklearn.feature_extraction.text- Support ML alogorithms from the dataset
#CountVectorizer-  Text preprocessing technique, commonly used natural language processing
#sklearn.model_selection- cross validation our model
#MultinomialNB[NB:naive bayes] - classification 
#pickle - is used to serializing and de-serializing

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, render_template, request
import pickle

# Load the dataset
data = pd.read_csv("Language.csv")
print(data.head())

# Explore class distribution
print(data["Language"].value_counts())

# Split the data into features (X) and target variable (y)
x = np.array(data["Text"])
y = np.array(data["Language"])
 
# Convert text data to numerical features using CountVectorizer
cv = CountVectorizer()
#fit_trasform - used to fit data into model
X = cv.fit_transform(x)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train a Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the trained model to a file using pickle
with open("language_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Flask application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_text = request.form['user_input']
        
        # Transform user input using the same CountVectorizer
        user_data = cv.transform([user_text])

        # Make predictions
        output = model.predict(user_data)

        return render_template('index.html', prediction_text='Predicted Language: {}'.format(output[0]))

if __name__ == '__main__':
    app.run(debug=True)
