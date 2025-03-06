import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import plot_tree
from matplotlib.colors import ListedColormap

# 1. Загрузка данных
dataset = pd.read_csv('Life_Success.csv')
X = dataset.iloc[:, :-1].values  # Признаки
y = dataset.iloc[:, -1].values   # Целевая переменная: Успешная_жизнь
feature_names = dataset.columns[:-1].tolist()

# 2. Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("Обучающие признаки (первые 5):\n", X_train[:5])

# 3. Обучение случайного леса
classifier = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
classifier.fit(X_train, y_train)

# 4. Предсказание для нового примера
new_example = np.array([[30, 1, 1, 1, 1, 1, 0, 150]])  # Возраст=30, высшее, работа высокооплачиваемая, доход 150 тыс. леев
new_pred = classifier.predict(new_example)
print("Предсказание для нового примера:", "Да" if new_pred[0] == 1 else "Нет")

# 5. Предсказание и оценка
y_pred = classifier.predict(X_test)

# Метрики
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
print("Матрица ошибок:\n", cm)
print("Точность модели:", accuracy)
print("Точность (Precision):", precision)
print("Полнота (Recall):", recall)
print("F1-оценка:", f1)

# 6. Визуализация (для двух признаков: Возраст и Доход)
X_set, y_set = X_test[:, [0, 7]], y_test  # Выбираем Возраст и Доход для 2D-графика
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.5),
                     np.arange(start=X_set[:, 1].min() - 10, stop=X_set[:, 1].max() + 10, step=5))
X_grid = np.zeros((X1.size, X.shape[1]))
X_grid[:, 0] = X1.ravel()  # Возраст
X_grid[:, 7] = X2.ravel()  # Доход
# Заполняем остальные признаки средними значениями из X_train
for i in [1, 2, 3, 4, 5, 6]:
    X_grid[:, i] = np.mean(X_train[:, i])
plt.contourf(X1, X2, classifier.predict(X_grid).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('blue', 'orange')))
plt.scatter(X_set[:, 0], X_set[:, 1], c=y_set, cmap=ListedColormap(('blue', 'orange')), label='Тестовые данные')
plt.title('Random Forest (Возраст vs Доход)')
plt.xlabel('Возраст')
plt.ylabel('Доход (тыс. леев)')
plt.legend()
plt.show()

# 7. Визуализация дерева решений (из случайного леса)
# Визуализируем одно дерево
plt.figure(figsize=(20,10))
plot_tree(classifier.estimators_[0],
          feature_names=feature_names,
          class_names=['Нет', 'Да'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Дерево решений из случайного леса")
plt.show()
