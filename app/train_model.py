# train_model.py
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import make_pipeline
# import joblib
# from sklearn.model_selection import train_test_split

# def train_and_save_model(dataset_path, save_path='health_assistant_model.joblib'):
#     # Load the dataset
#     df = pd.read_csv(dataset_path)

#     # Check for missing values and handle them if needed
#     df = df.dropna()

#     # Split the data into training and testing sets
#     X_train, _, y_train, _ = train_test_split(df['question'], df['focus_area'], test_size=0.2, random_state=42)

#     # Build a pipeline with TF-IDF vectorizer and Multinomial Naive Bayes classifier
#     model = make_pipeline(TfidfVectorizer(), MultinomialNB())

#     # Train the model
#     model.fit(X_train, y_train)

#     # Save the model
#     joblib.dump(model, save_path)

# def get_bot_response(user_input):
#  # aditional test 
#     df = pd.read_csv('G:\\Volume E\\Virtual_Health_Assistant(BE)\\dataset\\med_2.csv')

#     # Load the saved model
#     health_assistant_model = joblib.load('health_assistant_model.joblib')

#     # Make a prediction for focus area
#     predicted_focus_area = health_assistant_model.predict([user_input])

#     # Assuming your dataset has a column 'answer'
#     # You might need to preprocess and clean the answer text similarly to questions
#     # Extract the answer corresponding to the predicted focus area
#     predicted_answer = df[df['focus_area'] == predicted_focus_area[0]]['answer'].values[0]

#     return predicted_answer

# if __name__ == '__main__':
#     train_and_save_model('G:\\Volume E\\Virtual_Health_Assistant(BE)\\dataset\\med_2.csv')


# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.pipeline import make_pipeline
# import joblib

# def train_and_save_model(dataset_path, save_path='health_assistant_model.joblib'):
#     # Load the dataset
#     df = pd.read_csv(dataset_path)

#     # Check for missing values and handle them if needed
#     df.dropna(inplace=True)  # Drop rows with missing values
#     df['question'] = df['question'].apply(lambda x: x.lower())  # Lowercase text data

#     # Model Selection and Hyperparameter Tuning
#     X_train, _, y_train, _ = train_test_split(df['question'], df['focus_area'], test_size=0.2, random_state=42)
#     pipeline = make_pipeline(TfidfVectorizer(), RandomForestClassifier(random_state=42))
#     param_grid = {
#         'randomforestclassifier__n_estimators': [100, 200, 300],
#         'randomforestclassifier__max_depth': [None, 10, 20],
#         'tfidfvectorizer__max_features': [1000, 2000, 3000]
#     }
#     grid_search = GridSearchCV(pipeline, param_grid, cv=5)
#     grid_search.fit(X_train, y_train)

#     # Save the model
#     joblib.dump(grid_search.best_estimator_, save_path)

# def get_bot_response(user_input):
#     df = pd.read_csv('G:/Volume E/Virtual_Health_Assistant(BE)/dataset/med_2.csv')
#     health_assistant_model = joblib.load('health_assistant_model.joblib')
#     predicted_focus_area = health_assistant_model.predict([user_input])

#     if not user_input or user_input.isspace():
#         return "Please provide a valid question."

#     # Check if the predicted focus area exists in the dataset
#     if predicted_focus_area[0] in df['focus_area'].values:
#         # Extract the answer corresponding to the predicted focus area
#         predicted_answer = df[df['focus_area'] == predicted_focus_area[0]]['answer'].values[0]
#         return predicted_answer
#     else:
#         return "No answer found for the predicted focus area."

# if __name__ == '__main__':
#     train_and_save_model('G:/Volume E/Virtual_Health_Assistant(BE)/dataset/med_2.csv')

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
from fuzzywuzzy import process

def train_and_save_model(dataset_path, save_path='health_assistant_model.joblib'):
    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Check for missing values and handle them if needed
    df.dropna(inplace=True)  # Drop rows with missing values
    df['question'] = df['question'].apply(lambda x: x.lower())  # Lowercase text data

    # Model Selection and Hyperparameter Tuning
    X_train, X_test, y_train, y_test = train_test_split(df['question'], df['focus_area'], test_size=0.2, random_state=42)
    pipeline = make_pipeline(TfidfVectorizer(), RandomForestClassifier(random_state=42))
    param_grid = {
        'randomforestclassifier__n_estimators': [100, 200, 300],
        'randomforestclassifier__max_depth': [None, 10, 20],
        'tfidfvectorizer__max_features': [1000, 2000, 3000]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Best parameters
    print("Best parameters:", grid_search.best_params_)

    # Make predictions on the test set
    predictions = grid_search.predict(X_test)

    # Evaluate accuracy and other metrics
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy: {accuracy:.2f}')

    # Display classification report
    print(classification_report(y_test, predictions))

    # Save the model
    joblib.dump(grid_search.best_estimator_, save_path)

    return grid_search.best_estimator_

def get_bot_response(user_input, model_path='health_assistant_model.joblib', dataset_path='https://medcsv.s3.ap-south-1.amazonaws.com/uploads/med_2.csv'):
    df = pd.read_csv(dataset_path)
    health_assistant_model = joblib.load(model_path)

    if not user_input or user_input.isspace():
        return "Please provide a valid question."
    else:
        # Preprocess user input
        user_input = user_input.lower()  # Convert to lowercase

        # Find the most similar question in the dataset
        similar_question = process.extractOne(user_input, df['question'])

        if similar_question[1] < 90:  # Adjust similarity threshold as needed
            return "Sorry, I couldn't understand your question. Please try again or ask a different question."
        else:
            # Find the corresponding focus area for the similar question
            predicted_focus_area = df.loc[df['question'] == similar_question[0], 'focus_area'].values[0]

            # Check if the predicted focus area exists in the dataset
            if predicted_focus_area in df['focus_area'].values:
                # Extract the answer corresponding to the predicted focus area
                predicted_answer = df[df['focus_area'] == predicted_focus_area]['answer'].values[0]
                # Return the predicted answer
                return predicted_answer
            else:
                # If focus area not found, return message
                return "No answer found for the predicted focus area."

if __name__ == '__main__':
    model = train_and_save_model('https://medcsv.s3.ap-south-1.amazonaws.com/uploads/med_2.csv', 'health_assistant_model.joblib')

