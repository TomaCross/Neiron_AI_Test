import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
x = np.array([
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1]
])
y = np.array([[0, 0, 1, 1]]).T
np.random.seed(1)
synapric_weights_0 = 2 * np.random.random((3, 4)) - 1
synapric_weights_1 = 2 * np.random.random((4, 1)) - 1

for iteration in range(60000):
    layer_0 = x
    layer_1 = sigmoid(np.dot(layer_0, synapric_weights_0))
    layer_2 = sigmoid(np.dot(layer_1, synapric_weights_1))
    layer_2_error = y - layer_2

    if (iteration % 10000) == 0:
        print('Ошибка: ' + str(np.abs(layer_2_error)))
    layer_2_delta = layer_2_error * sigmoid_derivative(layer_2)
    layer_1_error = layer_2_delta.dot(synapric_weights_1.T)
    layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)

    synapric_weights_1 += layer_1.T.dot(layer_2_delta)
    synapric_weights_0 += layer_0.T.dot(layer_1_delta)
print('\nРезлутат после обучения: ')
print(np.round(layer_2, 2))

np.save('Brain_0.npy', synapric_weights_0)
np.save('Brain_1.npy', synapric_weights_1)
print('Сохранено')