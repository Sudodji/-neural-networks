# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 20:29:37 2021
@author: AM4
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data.csv')

# возьмем первые 100 строк, 4-й столбец 
y = df.iloc[0:100, 4].values
# так как ответы у нас строки - нужно перейти к численным значениям
y = np.where(y == "Iris-setosa", 1, 0).reshape(-1,1) # reshape нужен для матричных операций

# возьмем два признака, чтобы было удобне визуализировать задачу
X = df.iloc[0:100, [0, 2]].values

# добавим фиктивный признак для удобства матричных вычислений
X = np.concatenate([np.ones((len(X),1)), X], axis=1)

# зададим функцию активации - сигмоида
def sigmoid(y):
    return 1 / (1 + np.exp(-y))

# нам понадобится производная от сигмоиды при вычислении градиента
def derivative_sigmoid(y):
    return sigmoid(y) * (1 - sigmoid(y))

# функция активации softmax
def softmax(y):
    exp_y = np.exp(y - np.max(y, axis=1, keepdims=True))
    return exp_y / np.sum(exp_y, axis=1, keepdims=True)

# производная softmax необходима для вычисления градиента
def derivative_softmax(y):
    s = softmax(y)
    return s * (1 - s)

# прямой проход
def feed_forward(x):
    input_ = x
    hidden_ = sigmoid(np.dot(input_, weights[0]))
    output_ = softmax(np.dot(hidden_, weights[1]))  # использование softmax
    return [input_, hidden_, output_]

# функция обратного прохода для softmax и категориальной кросс-энтропии
def backward_softmax(learning_rate, target, net_output, layers):
    err = (target - net_output)
    for i in range(len(layers)-1, 0, -1):
        err_delta = err * derivative_softmax(layers[i])       
        err = np.dot(err_delta, weights[i - 1].T)
        dw = np.dot(layers[i - 1].T, err_delta)
        weights[i - 1] += learning_rate * dw

# функция обучения с использованием стохастического обучения
def stochastic_train(x_values, target, learning_rate):
    # перемешиваем данные
    indexes = np.random.permutation(len(x_values))
    x_values_shuffled = x_values[indexes]
    target_shuffled = target[indexes]
    
    # проходим по всем данным в случайном порядке
    for x_val, target_val in zip(x_values_shuffled, target_shuffled):
        # прямой проход
        output = feed_forward(x_val)
        
        # обратный проход
        backward_softmax(learning_rate, target_val, output[2], output)

# инициализация нейронной сети 
inputSize = X.shape[1] # количество входных сигналов равно количеству признаков задачи 
hiddenSizes = 5 # задаем число нейронов скрытого слоя 
outputSize = y.shape[1] # количество выходных сигналов равно количеству классов задачи

# инициализация весов
weights = [
    np.random.uniform(-2, 2, size=(inputSize, hiddenSizes)),  # веса скрытого слоя
    np.random.uniform(-2, 2, size=(hiddenSizes, outputSize))  # веса выходного слоя
]

# функция обучения чередует прямой и обратный проход
def train(x_values, target, learning_rate):
    output = feed_forward(x_values)
    backward_softmax(learning_rate, target, output[2], output)
    return None

# функция предсказания возвращает только выход последнего слоя
def predict(x_values):
    return feed_forward(x_values)[-1]

# задаем параметры обучения
iterations = 50
learning_rate = 0.01

# обучаем сеть
for i in range(iterations):
    train(X, y, learning_rate)
    
    if i % 10 == 0:
        # Считаем категориальную кросс-энтропию
        loss = -np.sum(y * np.log(predict(X) + 1e-15)) / len(y)
        print("На итерации: " + str(i) + ' || ' + "Средняя ошибка: " + str(loss))

# считаем ошибку на обучающей выборке
pr = predict(X)
accuracy = np.mean(np.argmax(pr, axis=1) == np.argmax(y, axis=1))
print("Точность на обучающей выборке:", accuracy)


# считаем ошибку на всей выборке
y = df.iloc[:, 4].values
y = np.where(y == "Iris-setosa", 1, 0).reshape(-1,1) 
X = df.iloc[:, [0, 2]].values
X = np.concatenate([np.ones((len(X),1)), X], axis=1)

pr = predict(X)
accuracy = np.mean(np.argmax(pr, axis=1) == np.argmax(y, axis=1))
print("Точность на всей выборке:", accuracy)

