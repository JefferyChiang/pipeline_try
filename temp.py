import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion
X_digits, y_digits = datasets.load_digits(return_X_y=True)
X_digits = pd.DataFrame(X_digits).iloc[:5,:5]
def a(X):
    X = pd.DataFrame(X)
    X = X*2
    return X
aF = FunctionTransformer(a)
union = FeatureUnion([("pca", PCA(n_components=5)),
                     ("af", aF)])

pipe = Pipeline([('un',union),
        ('aF',aF)])

b = pd.DataFrame(union.fit_transform(X_digits))
c = pipe.fit_transform(X_digits)