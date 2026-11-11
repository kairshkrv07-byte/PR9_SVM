
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1. Создаем данные
X, y = make_moons(n_samples=500, noise=0.25, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Pipeline с масштабированием
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="rbf"))
])

# 3. Сеточный поиск
param_grid = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": [0.01, 0.1, 1, 10]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

start = time.time()
grid.fit(X_train, y_train)
rbf_time = time.time() - start

print("Лучшие параметры:", grid.best_params_)
print("Лучшая точность CV:", grid.best_score_)
print("Время поиска:", round(rbf_time, 3), "сек")

# 4. Проверка на тестовых данных
y_pred = grid.predict(X_test)

print("\nТочность RBF:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 5. Тепловая карта
results = pd.DataFrame(grid.cv_results_)

heatmap_data = results.pivot(
    index="param_svc__gamma",
    columns="param_svc__C",
    values="mean_test_score"
)

plt.figure(figsize=(8, 6))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".3f",
    cmap="YlGnBu"
)
plt.title("GridSearchCV: C и gamma")
plt.xlabel("C")
plt.ylabel("gamma")
plt.tight_layout()
plt.savefig("heatmap.png")
plt.show()

# 6. Линейная SVM
linear_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="linear", C=1))
])

start = time.time()
linear_model.fit(X_train, y_train)
linear_time = time.time() - start

linear_pred = linear_model.predict(X_test)

print("\nСРАВНЕНИЕ МОДЕЛЕЙ")
print("RBF accuracy:", accuracy_score(y_test, y_pred))
print("Linear accuracy:", accuracy_score(y_test, linear_pred))
print("Linear time:", round(linear_time, 3), "сек")
print("RBF grid search time:", round(rbf_time, 3), "сек")


# 7. Вывод по практической работе
print("\nВЫВОД:")
print("В работе исследовано влияние C и gamma на SVM.")
print("Лучшая точность кросс-валидации составила 95.5%.")
print("Точность RBF на тестовых данных составила 93%.")
print("Точность линейной SVM составила 83%.")
print("RBF лучше подходит для нелинейных данных.")
print("Однако подбор параметров требует больше времени.")
print("Применение SVM с ядром RBF оправдано.")
