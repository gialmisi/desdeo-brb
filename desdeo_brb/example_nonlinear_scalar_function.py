"""Code that has been used to produce the numerical example in my master's thesis, Section 3.7"""
import numpy as np
import matplotlib.pyplot as plt

from brb import BRB


# function to be modelled
def f(x):
    return np.cos(np.sqrt(x))**2 + np.cos(x)**2

# lower and upper bounds of the variable
lb = 0
ub = 5

# create evenly distributed training data, this is the observed data
n_train = 1000
xs_train = np.linspace(lb, ub, n_train, endpoint=True).reshape(-1, 1)
ys_train = f(xs_train)

# create random evaluation points
n_eval = 200
xs = np.sort(np.random.uniform(lb, ub, (n_eval, 1)), axis=0)
ys = f(xs)

# create a brb model with given referential values
precedents = np.array([[0, 1, 2, 3, 4, 5]])
consequents = np.array([[0, 0.5, 1, 1.5, 2]])
# consequents = np.array([[0, 1, 2, 3, 4]])

# construct an initial BRB model
brb = BRB(precedents, consequents, f=f)
print("Before training")
print(brb)

# untrained predictions on evaluation data
res = brb.predict(xs)
ys_untrained = np.sum(
    res.consequents * res.consequent_belief_degrees, axis=1
)

# train the model
brb.train(xs_train, ys_train, brb._flatten_parameters(), use_de=False)

print("After training")
print(brb)

res = brb.predict(xs)
ys_trained = np.sum(
    res.consequents * res.consequent_belief_degrees, axis=1
)

plt.title(r"Trained vs untrained BRB system for predicting the non-linear function $f(x) = \cos(\sqrt{x})^2 + \cos(x)^2$")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(xs, ys, label="f(x) = y", ls="dotted")
plt.plot(xs, ys_untrained, label="Untrained", ls="--")
plt.plot(xs, ys_trained, label="Trained")

plt.ylim((0, 2.1))
plt.legend()

plt.show()
