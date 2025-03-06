# Импорт библиотек
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Загрузка данных
dataset = pd.read_csv('Customer_Survey.csv')
X = dataset.iloc[:, :-1].values  # Признаки: ЧасыОнлайн, Траты
y = dataset.iloc[:, -1].values   # Целевая переменная: КупилПремиум

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print("Обучающие признаки:\n", X_train)
print("Обучающие метки:\n", y_train)
print("Тестовые признаки:\n", X_test)
print("Тестовые метки:\n", y_test)

# Масштабирование признаков
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print("Масштабированные обучающие признаки:\n", X_train)
print("Масштабированные тестовые признаки:\n", X_test)

# Обучение модели Наивного Байеса на обучающей выборке
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Предсказание для нового примера
new_prediction = classifier.predict(sc.transform([[15, 250]]))
print("Предсказание для ЧасыОнлайн=15, Траты=250:", new_prediction)

# Предсказание результатов на тестовой выборке
y_pred = classifier.predict(X_test)
print("Предсказанные vs Реальные:\n", np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Вычисление метрик
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Вывод результатов
print("\nОценка модели:")
print("Матрица ошибок:\n", cm)
print("Точность (Accuracy): {:.2f}".format(accuracy))
print("Точность (Precision): {:.2f}".format(precision))
print("Полнота (Recall): {:.2f}".format(recall))
print("F1-мера: {:.2f}".format(f1))

# Оценка полученных результатов
print("\nАнализ результатов:")
if accuracy > 0.8:
    print("Точность модели высокая (более 80%), что говорит о хорошей общей производительности.")
else:
    print("Точность модели ниже ожидаемого (менее 80%), возможно, стоит попробовать другую модель или улучшить данные.")
if precision > 0.8:
    print("Precision высокий, модель хорошо определяет положительные случаи без лишних ложных срабатываний.")
else:
    print("Precision низкий, модель склонна к ложным положительным результатам.")
if recall > 0.8:
    print("Recall высокий, модель успешно находит большинство положительных случаев.")
else:
    print("Recall низкий, модель упускает значительную часть положительных случаев.")
if f1 > 0.8:
    print("F1-мера высокая, что указывает на сбалансированность между Precision и Recall.")
else:
    print("F1-мера низкая, модель несбалансирована по точности и полноте.")

# Визуализация результатов на обучающей выборке
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 5, stop=X_set[:, 0].max() + 5, step=0.25),
                     np.arange(start=X_set[:, 1].min() - 50, stop=X_set[:, 1].max() + 50, step=10))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('purple', 'yellow')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], color=ListedColormap(('purple', 'yellow'))(i), label=j)
plt.title('Наивный Байес (Обучающая выборка)')
plt.xlabel('Часы онлайн')
plt.ylabel('Траты ($)')
plt.legend()
plt.show()

# Визуализация результатов на тестовой выборке
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 5, stop=X_set[:, 0].max() + 5, step=0.25),
                     np.arange(start=X_set[:, 1].min() - 50, stop=X_set[:, 1].max() + 50, step=10))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('purple', 'yellow')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], color=ListedColormap(('purple', 'yellow'))(i), label=j)
plt.title('Наивный Байес (Тестовая выборка)')
plt.xlabel('Часы онлайн')
plt.ylabel('Траты ($)')
plt.legend()
plt.show()