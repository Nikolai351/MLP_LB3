import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Функция исключающее или
def xor(a, b):
    return a != b

# Создаем набор данных
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем модель MLP
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)

# Обучаем модель
mlp.fit(X_train, y_train)

# Проверяем точность модели на тестовых данных
print("Точность на тестовых данных:", mlp.score(X_test, y_test))
