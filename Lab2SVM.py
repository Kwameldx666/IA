# Импорт библиотек
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# 1. Загрузка данных
# Предположим, у нас есть файл 'Student_Data.csv' с колонками: ЧасыУчебы, Оценка, Стипендия
dataset = pd.read_csv('Student_Data.csv')
X = dataset.iloc[:, :-1].values  # Признаки: ЧасыУчебы, Оценка
y = dataset.iloc[:, -1].values   # Целевая переменная: Стипендия (0 или 1)

# 2. Разделение данных на обучающую и тестовую выборки
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Обучающие признаки:\n", X_train)
print("Тестовые признаки:\n", X_test)

# 3. Масштабирование признаков
# SVM чувствителен к масштабу данных, поэтому стандартизируем признаки
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # Вычисляем среднее и стандартное отклонение на обучающей выборке
X_test = sc.transform(X_test)       # Применяем те же параметры к тестовой выборке
print("Масштабированные обучающие признаки:\n", X_train)

# 4. Обучение модели SVM
# Используем SVM с линейным ядром (можно попробовать и другие: 'rbf', 'poly')
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=42)
classifier.fit(X_train, y_train)

# 5. Предсказание для нового примера
# Проверим, получит ли студент стипендию при 10 часах учебы и оценке 85
new_prediction = classifier.predict(sc.transform([[10, 85]]))
print("Предсказание для ЧасыУчебы=10, Оценка=85:", "Да" if new_prediction[0] == 1 else "Нет")

# 6. Предсказание на тестовой выборке
y_pred = classifier.predict(X_test)
print("Предсказанные vs Реальные:\n", np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

cm = confusion_matrix(y_test, y_pred)
print("Матрица ошибок:\n", cm)

# 2. Расчет precision, recall и f1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Точность (Precision):", precision)
print("Полнота (Recall):", recall)
print("F1-score:", f1)

# 3. Точность модели
accuracy = accuracy_score(y_test, y_pred)
print("Точность модели (Accuracy):", accuracy)


# 8. Визуализация обучающей выборки
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.1),
                     np.arange(start=X_set[:, 1].min() - 5, stop=X_set[:, 1].max() + 5, step=1))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('blue', 'orange')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], color=ListedColormap(('blue', 'orange'))(i), label=j)
plt.title('SVM (Обучающая выборка)')
plt.xlabel('Часы учебы')
plt.ylabel('Оценка')
plt.legend()
plt.show()

# 9. Визуализация тестовой выборки
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.1),
                     np.arange(start=X_set[:, 1].min() - 5, stop=X_set[:, 1].max() + 5, step=1))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('blue', 'orange')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], color=ListedColormap(('blue', 'orange'))(i), label=j)
plt.title('SVM (Тестовая выборка)')
plt.xlabel('Часы учебы')
plt.ylabel('Оценка')
plt.legend()
plt.show()