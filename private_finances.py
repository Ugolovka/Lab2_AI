import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


data1 = pd.DataFrame({"food": ["pasta", "doshirak", "draniki"],
                      "summa": [22, 2, 10]})
data2 = pd.DataFrame({"Activities": ["Museum visits", "Concerts", "ballet"],
                      "summa": [10, 140, 160]})
data3 = pd.DataFrame({"technik": ["Laptop", "Smartphone", "Television"],
                      "summa": [3700, 3800, 9000]})


def category(df, column_name):
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(df[[column_name]])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column_name]))
    result = pd.concat([encoded_df, df["summa"]], axis=1)
    result["importance"] = result["summa"].apply(lambda x: "важная" if x > 100 else "неважная")
    return result


proc1 = category(data1, "food")
proc2 = category(data2, "Activities")
proc3 = category(data3, "technik")


full_df = pd.concat([proc1, proc2, proc3], ignore_index=True)
print(full_df)


y = full_df["importance"]
X = full_df.drop(columns=["importance"])


clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y)


print("Важность признаков:")
for name, score in zip(X.columns, clf.feature_importances_):
    print(f"{name}: {score}")
