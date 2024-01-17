from joblib import dump
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier


face = datasets.load_iris(return_X_y=True)
y = face[1]
X = face[0]


clf_pipeline = [('scaling', MinMaxScaler()), ('clf', GradientBoostingClassifier())]
pipeline = Pipeline(clf_pipeline)

pipeline.fit(X, y)

dump(pipeline, './ml/face_dt_v1.joblib')