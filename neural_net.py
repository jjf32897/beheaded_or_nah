'''
Predicts character deaths in Game of Thrones using a neural network with stochastic gradient descent strategy

Highest achieved accuracy: 69.7%
'''

import brainz
import pandas as pd
from sklearn.preprocessing import Imputer, OneHotEncoder
from numpy import array

# make data frame with training data
theverybest = pd.read_csv("data/character-predictions_edited_onehot.csv")

# define which targets are the variables and which are the features
Y = theverybest["isAlive"].values
columns = ["male", "culture", "house", "isAliveMother", "isAliveFather", "isAliveHeir", "isAliveSpouse", "isMarried", "isNoble", "age", "boolDeadRelations", "popularity"]

features = theverybest[list(columns)].values

# fills in NaN entries with average values... is there a better way to do this?
tyrion = Imputer(missing_values="NaN", strategy="mean", axis=0)
X = tyrion.fit_transform(features)

# one-hot encoding
enc = OneHotEncoder(categorical_features=[1,2])
X = enc.fit_transform(X).toarray()

# make training dataset (first 1000)
training = []

for i in xrange(1000):
	# result array that holds expected output
	result = array([[0], [0]])

	# changes appropriate element of result array to 1 or 0 depending on whether or not they are alive
	result[Y[i]] = 1

	# separates the character's values into separate lists because that's how they want it I guess
	training.append((array([[elem] for elem in X[i]]), result))

# make test dataset
testing = []

for i in xrange(1000, len(X)):
	# dead or alive?
	result = Y[i]

	# separates the character's values into separate lists because that's how they want it I guess
	testing.append((array([[elem] for elem in X[i]]), result))

# instantiate network
net = brainz.Network([423, 30, 2]) # does changing to one output node affect anything?

# running
net.train(training, 30, 10, 3.0, test=testing)

