# 🚢 Titanic — Machine Learning from Disaster

🎯 **Мета:** передбачити, хто виживе на основі даних про пасажирів.

## 📄 Дані
- train.csv з Kaggle: https://www.kaggle.com/c/titanic

## 🔧 Методи
- Заповнення пропусків
- Кодування категорій
- Feature Engineering: FamilySize, IsAlone, Title
- Logistic Regression
- Підбір параметрів (GridSearchCV)
- Крос-валідація
- Графіки: важливість ознак, кореляційна матриця

## 📊 Результати
- Найкраще accuracy на крос-валідації: **82.4%**
- Параметри: `C=100, penalty=l2, solver=liblinear`

## 📂 Файли
- `titanic.ipynb` — код
- `titanic_model.joblib` — збережена модель
- `README.md` — опис проєкту

---