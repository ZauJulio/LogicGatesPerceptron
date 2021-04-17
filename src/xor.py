from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = [[0, 0], [0, 1], [1, 0], [1, 1]] * 1000
y = [0, 1, 1, 0] * 1000

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)


model = MLPClassifier(
    activation='tanh',
    solver='lbfgs',
    hidden_layer_sizes=(4),
    alpha=10,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test).tolist()

print(f"Validation set: {y_test[:20]}...")
print(f"Estimated  set: {y_pred[:20]}...")


# Average accuracy
print('Accuracy: %.8f %%' % (accuracy_score(y_test, y_pred)*100))
print("\Synaptic weights: \n", model.coefs_)
