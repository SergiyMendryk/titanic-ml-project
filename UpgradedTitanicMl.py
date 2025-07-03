# %%
# 1. Імпорти
# Стандартні
import numpy as np
import pandas as pd

# Візуалізація
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# %%
# 2. Завантаження даних
df_raw = pd.read_csv('/Users/sergiymendrik/Desktop/ІТ/ML Projects/Titanic - Machine Learning from Disaster/train.csv')

# %%
# 3. Первинний огляд
df_raw.shape
df_raw.columns
df_raw.info()
df_raw.isnull().sum()

# %%
# 4. Предобробка
def preprocess_data(df):
    df = df.copy()
    
    # Базове очищення
    df.drop(columns=['Cabin','Ticket'],errors = 'ignore', inplace = True)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Кодування статті
    df['Sex'] = df['Sex'].map({'male':0,'female':1})
    
    # Створення нових ознак
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Витяг заголовку з імені
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.',expand = False)
    df['Title'] = df['Title'].replace(['Mlle','Ms'], 'Miss')
    df['Title'] = df['Title'].replace(['Mme'],'Mrs')
    df['Title'] = df['Title'].replace(['Dr','Major','Col','Rev','Capt','Don','Sir','Countess','Lady','Jonkheer'],'Rare')
    
    # Кодування категорій 
    df = pd.get_dummies(df,columns=['Embarked','Title'],drop_first=True)
    
    # Видаляємо непотрібні
    df.drop(columns=['PassengerId','Name'],errors='ignore',inplace=True)
    
    return df

# %%
# 5. Побудова моделі
def train_and_evaluate(df):
    df_processed = preprocess_data(df)
    X = df_processed.drop('Survived',axis=1)
    y = df_processed['Survived']
    
    X_train,X_test,y_train,y_test = train_test_split(
        X,y, test_size=0.2,random_state=42,stratify=y
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train,y_train)
    
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test,y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test,y_pred))
    print("Classification report:\n", classification_report(y_test,y_pred))
    
    # 6. Оцінка моделі - графік матриці плутанини
    cm= confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

# %%
train_and_evaluate(df_raw)

# %%
# 7. Крос-валідація
def cross_validation(df, cv=5):
    df_processed = preprocess_data(df)
    X = df_processed.drop('Survived',axis=1)
    y = df_processed['Survived']
    
    model = LogisticRegression(max_iter=1000)
    scores = cross_val_score(model,X,y,cv=cv,scoring='accuracy')
    
    print(f"Кросс валідація ({cv}-fold):")
    print(f"Accuracy для кожної ітерації :", scores)
    for i , score in enumerate(scores,1):
            print(f" -Fold {i}: {score:.4f}")
    print(f"Середнє Accuracy: {scores.mean():.4f}")
    print(f"Стандартне відхилення: {scores.std():.4f}")

# %%
cross_validation(df_raw,cv=5)

# %%
# 8. Підбір параметрів
def grid_search_logreg(df):
    df_processed = preprocess_data(df)
    X = df_processed.drop('Survived',axis=1)
    y = df_processed['Survived']
    
    param_grid = {
        'C':[0.01,0.1,1,10,100],
        'penalty':['l2'],
        'solver':['liblinear','lbfgs']
    }
    
    model = LogisticRegression(max_iter=1000)
    grid = GridSearchCV(model,param_grid,cv=5,scoring='accuracy')
    grid.fit(X,y)
    print(f"Найкращі параметри:", grid.best_params_)    
    print(f"Найкращий accuracy: {grid.best_score_:.4f}")

# %%
grid_search_logreg(df_raw)

# %%
# 9. Важливість ознак та графіки
# (Цей розділ поки порожній, можна додати графіки важливості ознак, наприклад coef_ моделі або інші візуалізації)
