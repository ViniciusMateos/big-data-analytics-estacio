import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
# libs necessárias

# treinamento feito com validação cruzada
clf = DecisionTreeClassifier(max_depth=3, random_state=0)
iris = load_iris()
cross_val_score(clf, iris.data, iris.target, cv=10)

# dados carregados e o experimento executado para checarmos a possível média de acurácia do modelo treinado com o conjunto,
# vamos ao treinamento propriamente dito e à visualização do resultado.

clf.fit(iris.data, iris.target)
plot_tree(clf, filled=True)
plt.show() # Árvore de decisão resultante da implementação em Python.