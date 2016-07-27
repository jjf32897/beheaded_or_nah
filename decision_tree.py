from numpy import array
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import Imputer, OneHotEncoder
from sklearn.externals.six import StringIO

# make data frame with training data
theverybest = pd.read_csv("data/character-predictions_edited_onehot.csv")

# define which targets are the variables and which are the features
Y = theverybest["isAlive"].values
columns = ["male", "culture", "house", "isAliveMother", "isAliveFather", "isAliveHeir", "isAliveSpouse", "isMarried", "isNoble", "age", "boolDeadRelations", "popularity"]

features = theverybest[list(columns)].values

# one-hot encoding of cultures and houses
enc = OneHotEncoder()


tyrion = Imputer(missing_values="NaN", strategy="mean", axis=0)
X = tyrion.fit_transform(features)

clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(X, Y)

print clf.predict(array([1, 1, 30, 1, 1, 1, 1, 1, 1, 9, 1, 1]).reshape(1, -1))

# with open("dead.dot", "w") as f:
#     f = tree.export_graphviz(clf, out_file=f, feature_names=columns)
