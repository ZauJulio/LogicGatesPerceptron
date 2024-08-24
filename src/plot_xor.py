from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt

import numpy as np


# Set Data and Train Model
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 1000)
y = np.array([0, 1, 1, 0] * 1000)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

model = MLPClassifier(
    activation='tanh',
    solver='lbfgs',
    hidden_layer_sizes=(4),
    alpha=10,
)

model.fit(X_train, y_train)

score = model.score(X_test, y_test)

y_pred = model.predict(X_test)

# Plot classification
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                     np.arange(y_min, y_max, .02))


ax = plt.subplot(1, 1, 1)

ax.contourf(
    xx,
    yy,
    model.predict_proba(
        np.c_[xx.ravel(), yy.ravel()]
    )[:, 1].reshape(xx.shape),
    cmap=plt.cm.RdBu,
    alpha=.8
)

# Plot train data
ax.scatter(
    X_train[:, 0] + .01,
    X_train[:, 1] + .01,
    c=y_train,
    label="Train data*",
    cmap=ListedColormap(['#FF0000', '#0000FF']),
    edgecolors='black',
    s=25
)

# Plot test data
ax.scatter(
    X_test[:, 0] - .01,
    X_test[:, 1] - .01,
    c=y_test,
    label="Test data*",
    cmap=ListedColormap(['#FF0000', '#0000FF']),
    edgecolors='yellow',
    s=25
)

# Plot validation data
ax.scatter(
    X_test[:, 0] - .01,
    X_test[:, 1] + .01,
    c=y_pred,
    label="Validation data*",
    cmap=ListedColormap(['#FF0000', '#0000FF']),
    edgecolors='white',
    s=25
)
"""
* The points were slightly displaced to
facilitate viewing, but represent integer
values [0, 1].
"""

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_title("""
MLP - XOR Model | * The points were slightly displaced to
facilitate viewing, but represent integer values [0, 1]""")
ax.legend()
plt.show()
