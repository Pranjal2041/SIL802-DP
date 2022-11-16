import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from diffprivlib.models import GaussianNB

dataset = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)

epsilons = np.logspace(-2, 6, 50)
bounds = ([4.3, 2.0, 1.1, 0.1], [7.9, 4.4, 6.9, 2.5])
accuracy = list()

for epsilon in epsilons:
    clf = GaussianNB(bounds=bounds, epsilon=epsilon)
    clf.fit(X_train, y_train)
    
    accuracy.append(clf.score(X_test, y_test))


smooths = [accuracy[0]]
alpha = 0.2
for x in accuracy[1:]:
    smooths.append(x * alpha + (1-alpha) * smooths[-1])


plt.semilogx(epsilons, smooths, alpha = 0.4)
plt.semilogx(epsilons, accuracy)
plt.semilogx(epsilons, [accuracy[-1] for _ in range(len(accuracy))], alpha = 0.4, linestyle = 'dashed')

plt.title("Differentially private Naive Bayes accuracy")
plt.xlabel("epsilon")
plt.ylabel("Accuracy")
# Create a grid
plt.grid(linestyle = '--')
plt.legend(["Smoothed", "Unsmoothed", "Without Diff. Privacy"])
plt.savefig("NB_Plot.png")
plt.show()
