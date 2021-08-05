import pandas as pd  # Для работы с данными
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline  # используем пайплайны для удобства
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

data = pd.read_csv('adult_csv.csv')
selectedColumns = data[['age', 'workclass', 'education-num', 'fnlwgt']]
X = pd.get_dummies(selectedColumns, columns=['education-num', 'workclass'])

le = LabelEncoder()
y = le.fit(data['class'])
y = pd.Series(data=le.transform(data['class']))

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model = LogisticRegression()  # берем в качестве модели логистическую регресиию из scikit-learn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model.fit(X_train, y_train)  # обучаем на части датасета (train)
predictions = model.predict(X_test)

model.predict(X_test)  # получаем массив
model.predict_proba(X_test)
model.score(X_train, y_train)  # Получаем наш скор (точность предсказания) на обучающей и тестовой выборках
model.score(X_test, y_test)  # Получаем наш скор (точность предсказания) на обучающей и тестовой выборках

model2 = SVC()
model2.fit(X, y)
model.score(X_train, y_train)
model.score(X_test, y_test)