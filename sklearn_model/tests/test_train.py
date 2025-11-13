import unittest
import pandas as pd
from app.train import pipe

def test_pipeline_training():
    data = pd.DataFrame({'message': ['Hello', 'Win money now'], 'label':[0,1]})
    pipe.fit(data['message'], data['label'])
    assert pipe.score(data['message'], data['label']) > 0.5
