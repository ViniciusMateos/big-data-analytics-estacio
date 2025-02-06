import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# libs necessárias para implementar um classificador SVM

# importaremos o dataset que será explorado em nossos estudos: o dataset Íris,
# clássico no âmbito do Aprendizado de Máquina
# classifica tipos de flores da espécie Íris a partir de características como comprimento de pétala, largura da pétala, comprimento da sépala e largura da sépala (todas em cm)

from sklearn.datasets import load_iris
data = load_iris()
iris = pd.DataFrame(data['data'], columns=data.feature_names)
target = data.target

# instanciação simples do classificador
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC # observando que o classificador SVM é identificado no scikit-learn a partir do nome SVC - C de Classifier
svc = SVC(gamma='auto')

# A validação cruzada é um dos modos mais comuns de se treinar nossos modelos
# pois ela divide o conjunto de dados em (k-1)/k partições de treinamento
# 1/k de teste de maneira circular e interativa, tendo assim todas as 1/k possíveis partições, podendo ser testadas contra o resto.

#Testando o modelo 'svc' na nossa base 'iris'
cv_result = cross_val_score(svc, iris, target, cv=10, scoring='accuracy')
#Retorna a acurácia em porcentagem do nosso modelo
print('Acurácia com cross validation:', cv_result.mean()*100)

# vamos treinar nosso modelo com o dataset inteiro e tentar predizer um valor inédito.
svc.fit(iris, target)
#Prediz a que classe pertencerá a flor com sépala de comprimento 6.9 cm e de largura 2.8 cm, e com pétala de comprimento 6.1 cm e de largura 2.3 cm
svc.predict([[6.9,2.8,6.1,2.3]])

# visualizar nossos dados e os hiperplanos definidos pelo modelo
plt.scatter(iris['sepal length (cm)'], iris['petal width (cm)'], c=target)
plt.title('Iris')
plt.show()

#Provavelmente, criando 2 features novas no íris, o svm com 2 features terá mais sucesso, mas por enquanto usei só
#sepal length e petal width (os mais relevantes das 4 features já existentes)
x0_min, x0_max = iris['sepal length (cm)'].min(), iris['sepal length (cm)'].max()
x1_min, x1_max = iris['petal width (cm)'].min(), iris['petal width (cm)'].max()
w = x0_max - x0_min
h = x1_max - x1_min
x0, x1 = np.meshgrid(np.linspace(x0_min-.1*w, x0_max+.1*w, 300),
                        np.linspace(x1_min-.1*h, x1_max+.1*h, 300))
svc.fit(iris[['sepal length (cm)', 'petal width (cm)']], target)
ypred = svc.predict(np.c_[x0.reshape(-1, 1), x1.reshape(-1, 1)])
ypred = ypred.reshape(x0.shape)
plt.contourf(x0, x1, ypred)
plt.scatter(iris['sepal length (cm)'], iris['petal width (cm)'], c=target, s=64, edgecolors='k')
plt.title('Iris')
plt.show()