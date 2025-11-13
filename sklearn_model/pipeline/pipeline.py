from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

def build_pipeline():
    return Pipeline([
        ('vect', CountVectorizer()),
        ('clf', LogisticRegression())
    ])
