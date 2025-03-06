import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import export_text  # Для вывода правил

# 1. Загрузка данных
dataset = pd.read_csv('Student_Data.csv')
X = dataset.iloc[:, :-1].values  # Признаки: ЧасыУчебы, Оценка
y = dataset.iloc[:, -1].values  # Целевая переменная: Стипендия

# Проверка данных
print("Первые 5 строк данных:\n", dataset.head())
print("Уникальные значения в y:", np.unique(y))

# 2. Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Обучающие признаки (первые 5):\n", X_train[:5])
print("Тестовые признаки (первые 5):\n", X_test[:5])

# 3. Обучение модели дерева решений
classifier = DecisionTreeClassifier(max_depth=3, random_state=42)  # Ограничиваем глубину для простоты
classifier.fit(X_train, y_train)

# 4. Извлечение правил (опционально)
feature_names = ['ЧасыУчебы', 'Оценка']
rules = export_text(classifier, feature_names=feature_names)
print("\nСгенерированные правила дерева решений:\n", rules)

# 5. Применение модели к обучающей выборке
y_train_pred = classifier.predict(X_train)
print("\nПредсказанные vs Реальные (обучающая выборка):\n",
      np.concatenate((y_train_pred.reshape(len(y_train_pred), 1), y_train.reshape(len(y_train), 1)), 1))

# 6. Оценка на обучающей выборке
cm_train = confusion_matrix(y_train, y_train_pred)
accuracy_train = accuracy_score(y_train, y_train_pred)
precision_train = precision_score(y_train, y_train_pred, zero_division=0)
recall_train = recall_score(y_train, y_train_pred, zero_division=0)
f1_train = f1_score(y_train, y_train_pred, zero_division=0)
print("\nОценка на обучающей выборке:")
print("Матрица ошибок:\n", cm_train)
print(f"Точность (Accuracy): {accuracy_train:.2f}")
print(f"Точность (Precision): {precision_train:.2f}")
print(f"Полнота (Recall): {recall_train:.2f}")
print(f"F1-мера: {f1_train:.2f}")

# 7. Применение модели к тестовой выборке
y_test_pred = classifier.predict(X_test)
print("\nПредсказанные vs Реальные (тестовая выборка):\n",
      np.concatenate((y_test_pred.reshape(len(y_test_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# 8. Оценка на тестовой выборке
cm_test = confusion_matrix(y_test, y_test_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred, zero_division=0)
recall_test = recall_score(y_test, y_test_pred, zero_division=0)
f1_test = f1_score(y_test, y_test_pred, zero_division=0)
print("\nОценка на тестовой выборке:")
print("Матрица ошибок:\n", cm_test)
print(f"Точность (Accuracy): {accuracy_test:.2f}")
print(f"Точность (Precision): {precision_test:.2f}")
print(f"Полнота (Recall): {recall_test:.2f}")
print(f"F1-мера: {f1_test:.2f}")

# 9. Предсказание для нового примера
new_example = np.array([[10, 85]])
new_pred = classifier.predict(new_example)
print("\nПредсказание для ЧасыУчебы=10, Оценка=85:", "Да" if new_pred[0] == 1 else "Нет")

# 10. Визуализация дерева решений
plt.figure(figsize=(12,8))
plot_tree(classifier, feature_names=feature_names, class_names=['Нет', 'Да'], filled=True, rounded=True, fontsize=12)
plt.title("Дерево решений для модели")
plt.show()
