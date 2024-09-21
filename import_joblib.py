import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# importing dataset
iris = load_iris()

# have the feature name as -
# print(iris.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
X, y = iris.data, iris.target

# print(iris.target_names)
# ['setosa' 'versicolor' 'virginica']

model = RandomForestClassifier()  # obj. creation
model.fit(X, y)                   # training model

# this will save and create a file for this model using JOBLIB
joblib.dump(model, 'model.joblib')