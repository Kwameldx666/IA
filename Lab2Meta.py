import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from matplotlib.colors import ListedColormap

# 1. Загрузка данных
dataset = pd.read_csv('Business_Success_Lei.csv')
X = dataset.iloc[:, :-1].values  # Признаки
y = dataset.iloc[:, -1].values   # Целевая переменная: Успешный_бизнес
feature_names = dataset.columns[:-1].tolist()

# Проверка данных
print("Первые 5 строк данных:\n", dataset.head())
print("Уникальные значения в y:", np.unique(y))

# 2. Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Обучающие признаки (первые 5):\n", X_train[:5])

# 3. Обучение модели AdaBoost
base_classifier = DecisionTreeClassifier(max_depth=1, random_state=42)  # Слабый классификатор
classifier = AdaBoostClassifier(estimator=base_classifier, n_estimators=30, random_state=42)
classifier.fit(X_train, y_train)

# 4. Предсказание для нового примера
new_example = np.array([[35, 1, 10, 120, 1, 1, 1]])  # 35 лет, образование, опыт, 120 тыс. леев, команда, риск, идея
new_pred = classifier.predict(new_example)
print("Предсказание: Успешный бизнес (Возраст=35, Инвестиции=120 тыс. леев):", "Да" if new_pred[0] == 1 else "Нет")

# 5. Предсказание и оценка
y_pred = classifier.predict(X_test)

# Вычисление метрик
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# Вывод результатов
print("\nОценка модели:")
print("Матрица ошибок:\n", cm)
print(f"Точность (Accuracy): {accuracy:.2f}")
print(f"Точность (Precision): {precision:.2f}")
print(f"Полнота (Recall): {recall:.2f}")
print(f"F1-мера: {f1:.2f}")

# Интерпретация результатов
print("\nИнтерпретация результатов:")
print(f"Из {len(y_test)} тестовых примеров:")
print(f" - Правильно предсказано 'Неуспешный бизнес' (TN): {cm[0, 0]}")
print(f" - Ошибочно предсказано 'Успешный' для 'Неуспешного' (FP): {cm[0, 1]}")
print(f" - Ошибочно предсказано 'Неуспешный' для 'Успешного' (FN): {cm[1, 0]}")
print(f" - Правильно предсказано 'Успешный бизнес' (TP): {cm[1, 1]}")
if accuracy > 0.8:
    print("Точность высокая (>80%), модель хорошо справляется с задачей.")
else:
    print("Точность ниже 80%, возможно, стоит улучшить модель или данные.")
if precision > 0.8:
    print("Precision высокий, ложных срабатываний мало.")
else:
    print("Precision низкий, много ложных предсказаний 'Успешный бизнес'.")
if recall > 0.8:
    print("Recall высокий, модель находит большинство успешных бизнесов.")
else:
    print("Recall низкий, модель пропускает много успешных бизнесов.")
if f1 > 0.8:
    print("F1-мера высокая, модель сбалансирована.")
else:
    print("F1-мера низкая, есть проблемы с балансом Precision и Recall.")

# 6. Важность признаков
importances = classifier.feature_importances_
for name, importance in zip(feature_names, importances):
    print(f"Важность {name}: {importance:.3f}")

# 7. Визуализация (Опыт_работы_лет vs Инвестиции_тыс_леев)
X_set, y_set = X_test[:, [2, 3]], y_test  # Выбираем Опыт и Инвестиции
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.5),
                     np.arange(start=X_set[:, 1].min() - 10, stop=X_set[:, 1].max() + 10, step=5))
X_grid = np.zeros((X1.size, X.shape[1]))
X_grid[:, 2] = X1.ravel()  # Опыт_работы_лет
X_grid[:, 3] = X2.ravel()  # Инвестиции_тыс_леев
for i in [0, 1, 4, 5, 6]:
    X_grid[:, i] = np.mean(X_train[:, i])  # Средние значения для остальных
plt.contourf(X1, X2, classifier.predict(X_grid).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('blue', 'orange')))
plt.scatter(X_set[:, 0], X_set[:, 1], c=y_set, cmap=ListedColormap(('blue', 'orange')), label='Тестовые данные')
plt.title('AdaBoost: Успешный бизнес (Опыт работы vs Инвестиции)')
plt.xlabel('Опыт работы (лет)')
plt.ylabel('Инвестиции (тыс. леев)')
plt.legend()
plt.show()