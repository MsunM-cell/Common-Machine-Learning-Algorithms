# import library
from sklearn.linear_model import LogisticRegression
import pandas as pd

# assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset

# create logistic regression object

model = LogisticRegression()

path = 'data.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])

cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]

# Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)

# equation coefficient and intercept
print('Coefficient: \n', model.coef_)
print('Intercept: \n', model.intercept_)

# Predict Output
# predicted = model.predict(x_test)
