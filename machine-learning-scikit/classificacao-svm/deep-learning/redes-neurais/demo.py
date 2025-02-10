from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

# os conjuntos de dígitos escritos à mão do MNIST 
# (Modified National Institute of Standards and Technology)

# nossa rede neural vai aprender a reconhecer o padrão de escrita de números.

# conjunto já vem no formato de treinamento e teste
# Cada imagem do MNIST tem dimensões 28x28x1, mas, para a rede neural, vamos precisar chapar a imagem em 28x28=784 pixels.

print('[INFO] accessing MNIST...')
((trainX, trainY), (testX, testY)) = mnist.load_data()


# normalizaremos os dados para que fiquem entre 0 e 1, e faremos isso dividindo o conjunto por 255 (valor máximo de um pixel)
trainX = trainX.reshape((trainX.shape[0], 28 * 28 * 1))
testX = testX.reshape((testX.shape[0], 28 * 28 * 1))
trainX = trainX.astype('float32') / 255.0
testX = testX.astype('float32') / 255.0


#  para adequar a última camada, a de saída, vamos binarizar a classe
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# LabelBinarizer faz com o que o resultado da classe se torne binário
# ao invés de lidarmos com a classe de valor 5, passaremos a lidar com 0000100000.
# Isso é importante, pois a camada final deve ter tamanho proporcional às possibilidades de resultados esperados

# Essa prática é muito comum em problemas de classificação multiclasse, 
# mas existe uma categoria que é a de multirótulos, em que os resultados podem ser de mais de uma classe ao mesmo tempo
# Por exemplo, categorias de autores de livros (Terror, Ficção Científica, Romance) admitem que um livro seja de Terror e Ficção Científica, tendo a binarização 110.
# Para cada caso como esse, existe uma função de custo diferente.


# definir a arquitetura da nossa rede neural.
# Com a ajuda do Keras, isso pode ser feito de maneira simples, adicionando uma camada atrás da outra em sequência,

model = Sequential()
model.add(Dense(256, input_shape=(784,), activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

#  a arquitetura seguirá este formato:

# Uma camada de entrada de 784 nós, um para cada pixel da imagem em questão, que se conectará a uma camada oculta densa de 256 nós pela função de ativação da sigmoide.
# Depois, a primeira camada oculta se conectará à segunda, de 128 nós, também por sigmoide.
# Esta se conectará à última camada de predição com 10 nós conectados a partir da Softmax. São 10 nós, porque temos 10 possíveis dígitos.

# Para treinar o modelo, vamos usar como otimizador do gradiente o SGD, aquele baseado no gradiente descendente, e com taxa de aprendizado 0.01
# Também faremos uso da métrica de acurácia para acompanhar o modelo.

# Para calcular a perda ou função de custo, vamos usar a entropia cruzada categórica (categorical_crossentropy), que é a mais utilizada em problemas de classificação.

# para as épocas da nossa rede, vamos escolher 100 épocas, ou seja, a rede tem 100 iterações para convergir e apreender, e vamos apresentar lotes de 128 imagens cada por iteração

sgd = SGD(0.01)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
H = model.fit(trainX, trainY, validation_data=(testX, testY),epochs=100, batch_size=128)


# classification_report, uma função do sklearn que compara os valores preditos com os reais, passados como argumentos.

predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1),
    target_names=[str(x) for x in lb.classes_]))

# resultado disso será um relatório de classificação.

# Como esse problema é multiclasse, além de mostrar a acurácia geral da classificação, o relatório apresentará o resultado de cada classe possível,


#resultado

#               precision    recall  f1-score   support

#            0       0.94      0.98      0.96       980
#            1       0.97      0.97      0.97      1135
#            2       0.93      0.89      0.91      1032
#            3       0.90      0.91      0.91      1010
#            4       0.92      0.93      0.93       982
#            5       0.91      0.86      0.88       892
#            6       0.93      0.95      0.94       958
#            7       0.93      0.92      0.92      1028
#            8       0.89      0.89      0.89       974
#            9       0.91      0.90      0.90      1009

#     accuracy                           0.92     10000
#    macro avg       0.92      0.92      0.92     10000
# weighted avg       0.92      0.92      0.92     10000

# Para interpretarmos esse relatório, tenha em mente que cada linha da matriz principal é uma possível classe,
# cada coluna é uma métrica de acompanhamento (no caso, precisão, recall e medida F1)
# finalizando com suporte (ou cobertura, ou seja, quantos casos foram cobertos pelas métricas escolhidas).

#Acurácia geral

# Média de acertos do modelo ao tentar prever o alvo do problema.
# É classificada como geral, pois contabiliza o número de acertos sem levar em consideração a ponderação entre possíveis classes.


#Média macro de acurácia

# Comparação a nível macro de acurácia para cada classe, feita sem considerar a distribuição da classe em relação às demais.


#Média ponderada

# Métrica que leva em consideração a distribuição das observações por classe em relação às demais.

####################################################

# As métricas macro, micro e ponderada são importantíssimas para conjuntos desbalanceados, pois, se tomarmos a métrica de modo generalista, isso pode dissuadir nossa avaliação de modelo,
# porque pode ser que o modelo tenha acertado tudo de algumas classes e pouco de outras, e essas métricas diferenciadas nos ajudam a encontrar esses detalhes.

#Em nossa análise, podemos perceber que o modelo tem mais facilidade em encontrar o dígito 1 do que outros dígitos e mais dificuldades de identificar os dígitos 8 e 5.
# De modo geral, a acurácia de 92% não é ruim, significando que, em cerca de 9 a cada 10 tentativas, a rede acerta
# Se analisarmos as métricas diferenciadas, podemos ver que, de maneira geral, os acertos entre classes estão bem equilibrados.

#####################################################

# com o código, a seguir, podemos ver como a rede evoluiu até chegar a essas métricas, ou seja, como a função de custo foi sendo otimizada e a acurácia foi subindo.

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 100), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, 100), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, 100), H.history['accuracy'], label='train_acc')
plt.plot(np.arange(0, 100), H.history['val_accuracy'], label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()

# o resultado da função de custo diminui à medida que as épocas passam, e a acurácia aumenta, o que é bastante intuitivo, pois a rede está aprendendo com o passar das épocas (iterações).
