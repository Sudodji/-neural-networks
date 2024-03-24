import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import random

df = pd.read_csv('data2.csv')

class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_size, hidden_sizes, out_size):
        super(MultiLayerPerceptron, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(in_size, hidden_sizes[0])])
        # Добавляем скрытые слои
        for k in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[k], hidden_sizes[k+1]))
        self.output = nn.Linear(hidden_sizes[-1], out_size)
    
    def forward(self, x):
        # Прямой проход через скрытые слои с активацией ReLU
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        # Прямой проход через выходной слой без активации
        x = self.output(x)
        return x

# Возьмем первую строку данных, чтобы определить размеры входного и выходного слоев
sample_row = df.iloc[0]
input_size = len(sample_row) - 1  # Размер входного слоя (количество признаков)
output_size = len(df.iloc[:, -1].unique())  # Размер выходного слоя (количество классов)

# Теперь определим размеры скрытых слоев
# В качестве примера, возьмем случайное количество скрытых слоев от 1 до 3
num_hidden_layers = random.randint(1, 3)
hidden_sizes = [random.randint(10, 50) for _ in range(num_hidden_layers)]

model_relu = MultiLayerPerceptron(input_size, hidden_sizes, output_size)

# Создаем модель с функцией активации Sigmoid
model_sigmoid = MultiLayerPerceptron(input_size, hidden_sizes, output_size)

# Определяем функцию потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer_relu = torch.optim.SGD(model_relu.parameters(), lr=0.01)
optimizer_sigmoid = torch.optim.SGD(model_sigmoid.parameters(), lr=0.01)

# Обучающие данные и метки
inputs = torch.randn(100, input_size)
labels = torch.randint(0, output_size, (100,))

# Обучаем модели
num_epochs = 100
for epoch in range(num_epochs):
    # Обучаем модель с функцией активации ReLU
    model_relu.train()
    optimizer_relu.zero_grad()
    outputs = model_relu(inputs)
    loss_relu = criterion(outputs, labels)
    loss_relu.backward()
    optimizer_relu.step()
    
    # Обучаем модель с функцией активации Sigmoid
    model_sigmoid.train()
    optimizer_sigmoid.zero_grad()
    outputs = model_sigmoid(inputs)
    loss_sigmoid = criterion(outputs, labels)
    loss_sigmoid.backward()
    optimizer_sigmoid.step()

    # Выводим статистику
    if (epoch+1) % 10 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss ReLU: {loss_relu.item():.4f}, Loss Sigmoid: {loss_sigmoid.item():.4f}')
