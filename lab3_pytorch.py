import torch
import torch.nn as nn
import torch.nn.functional as F

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

# Создаем модель с функцией активации ReLU
input_size = 10
hidden_sizes = [20, 30, 40]
output_size = 5
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
