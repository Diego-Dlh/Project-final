import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import mlflow

# Cargar dataset
df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])
X = df['message']
y = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear pipeline
pipe = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', LogisticRegression())
])

pipe.fit(X_train, y_train)
accuracy = pipe.score(X_test, y_test)

joblib.dump(pipe, 'spam_model.pkl')

# Log en MLflow
mlflow.start_run()
mlflow.log_param("model_type", "LogisticRegression")
mlflow.log_metric("accuracy", accuracy)
mlflow.sklearn.log_model(pipe, "model")
mlflow.end_run()
print(f"Accuracy: {accuracy}")
