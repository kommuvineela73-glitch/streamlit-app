import pickle
from sklearn.linear_model import LinearRegression
import numpy as np

# Sample marks → grade
X = np.array([[50],[60],[70],[80],[90]])
y = np.array([50,60,70,80,90])

model = LinearRegression()
model.fit(X, y)

# Save the trained model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)