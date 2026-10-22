import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
loaded_weights_0 = np.load('Brain_0.npy')
loaded_weights_1 = np.load('Brain_1.npy')

new_cust = np.array([[0, 1, 1]])
layer_1 = sigmoid(np.dot(new_cust, loaded_weights_0))
final = sigmoid(np.dot(layer_1, loaded_weights_1))

print('=' * 40)
print('Ответ буз обучения: ')
print('Чистый резултат: ', final)
print('Округлённый:', np.round(final, 2))
print('=' * 40)