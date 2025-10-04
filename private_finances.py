import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Исходные данные
data1 = pd.DataFrame({
    "food": ["pasta", "doshirak", "draniki"], # Категория еда + сумма
    "summa": [22, 2, 10]
})
data2 = pd.DataFrame({
    "Activities": ["Museum visits", "Concerts", "ballet"], # Категория развлечения + сумма
    "summa": [10, 140, 160]
    })
data3 = pd.DataFrame({
    "technik": ["Laptop", "Smartphone", "Television"], # Категория техника + сумма
    "summa": [3700, 3800, 9000]
    })

# Кодирование категориальных признаков и классификации важности
def category(df, column_name):
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(df[[column_name]])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column_name]))
    result = pd.concat([encoded_df, df["summa"]], axis=1)
    result["importance"] = result["summa"].apply(lambda x: "важная" if x > 100 else "неважная")
    return result

# Преобразование исходников
proc1 = category(data1, "food")
proc2 = category(data2, "Activities")
proc3 = category(data3, "technik")

full_df = pd.concat([proc1, proc2, proc3], ignore_index=True) # Объединение данных
print(full_df)

# Подготовка целевой переменной и признаков
y = full_df["importance"]
X = full_df.drop(columns=["importance"])

# Обучение дерева решений
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y)

# Вывод важности признаков
print("Важность признаков:")
for name, score in zip(X.columns, clf.feature_importances_):
    print(f"{name}: {score}")

# Визуализация дерева решений
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True, rounded=True)
plt.title("Дерево решений")
plt.show()

# Разделение данных на обучающие и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

clf = DecisionTreeClassifier(max_depth=3, random_state=1) # Обучение дерева
clf.fit(X_train, y_train)

# Оценка модели
pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, target_names=["важная", "неважная"]))

# Создание новых данных
all_features = X.columns.tolist() # Список всех признаков

new_data = pd.DataFrame({col: [0] for col in all_features}) # Все признаки = 0
new_data["food_draniki"] = 1 # Активируем категорию
new_data["summa"] = 45       # Устанавливаем занчение суммы


# Предсказание
pred = clf.predict(new_data)[0]
print("Предсказанная категория:", pred)



