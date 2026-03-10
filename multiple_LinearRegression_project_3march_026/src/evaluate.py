# ce n'est pas parceque un model a appris que cela veux dire que il est bon
import joblib
import math

from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import *

#load data to get feature_index

from visualization import (
plot_regression_plane_3d,
plot_simple_regression_feature,
plot_multiple_model_projection
)
# load model
model = joblib.load(MODEL_FILE)
# load test set
X_test = joblib.load(X_TEST_FILE)
y_test = joblib.load(Y_TEST_FILE)
X_test_transform = joblib.load(X_TEST_TRANSFORM)
X_Train = joblib.load(X_TRAIN_FILE)
Y_Train = joblib.load(Y_TRAIN_FILE)
X_TRAIN_SCALED=joblib.load(X_TRAIN_SCALED)
print(type(X_test))
print(type(y_test))
print(X_test.shape)
print(y_test.shape)
#load scaler
scale = joblib.load(SCALES_FILE)
#load Final DF
df_final = joblib.load(FINAL_DF)
df_feature_index = df_final.columns.get_loc(FEATURES_TO_USE_1[0]) #
df_feature_index_multiple_1 = df_final.columns.get_loc(FEATURES_TO_USE_1[0]) #
print(df_feature_index) # me renvoie 0 donc ma colonne 1
print(FEATURES_TO_USE_1[0]) # me renvoie ENGINESIZE

#prediction
y_pred = model.predict(X_test_transform)
print(f"prediction : {y_pred}")


# les metriques
MAE = mean_absolute_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
RMSE = math.sqrt(MSE)
R2 = r2_score(y_test, y_pred)

print(f"MAE: {MAE}")
print(f"MSE: {MSE}")
print(f"RMSE: {RMSE}")
print(f"R2: {R2}")


# plot_simple_regression_feature(X_Train,Y_Train,df_feature_index,FEATURES_TO_USE_1[0])
#plot_multiple_model_projection(X_TRAIN_SCALED,Y_Train,model,FEATURES_TO_USE_1,df_feature_index_multiple_1)
plot_regression_plane_3d(X_test_transform,y_test,model)
"""
Traceback (most recent call last):
  File "/Users/ermanw/Desktop/AI_ENGINEER/MACHINELEARNINGWITHPYTHON/projects_multiple_linearRegression/multiple_LinearRegression_project_3march_026/src/evaluate.py", line 55, in <module>
    plot_simple_regression_feature(X_Train,Y_Train,df_feature_index,FEATURES_TO_USE_2[0])
  File "/Users/ermanw/Desktop/AI_ENGINEER/MACHINELEARNINGWITHPYTHON/projects_multiple_linearRegression/multiple_LinearRegression_project_3march_026/src/visualization.py", line 59, in plot_simple_regression_feature
    model.fit(X_feature.reshape(-1,1),y_train)
              ^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/envs/.ml_projects_env/lib/python3.11/site-packages/pandas/core/generic.py", line 6206, in __getattr__
    return object.__getattribute__(self, name)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'Series' object has no attribute 'reshape'. Did you mean: 'shape'?



🔥 Parfait. Là l’erreur est très claire.

Regarde :

```
AttributeError: 'Series' object has no attribute 'reshape'
```

Pourquoi ?

Parce que :

```python
X_feature = X_train.iloc[:, df_feature_index]
```

retourne une **pandas Series**, pas un numpy array.

Et `.reshape()` existe seulement pour numpy.

---

# ❌ Mauvaise idée

```python
X_feature_csv = X_feature.to_csv()
```

Ça n’a aucun sens ici 😅

`to_csv()` transforme tes données en texte CSV.

Tu ne veux pas un fichier texte.
Tu veux juste un array numpy.

---

# ✅ Solution propre

Il faut convertir la Series en numpy array :

```python
X_feature = X_train.iloc[:, df_feature_index].values
```

ou

```python
X_feature = X_train.iloc[:, df_feature_index].to_numpy()
```

Puis :

```python
model.fit(X_feature.reshape(-1,1), y_train)
```

Et ça marchera.

---

# 🎯 Version finale correcte

```python
def plot_simple_regression_feature(X_train, y_train, feature_index, feature_name):

    X_feature = X_train.iloc[:, feature_index].to_numpy()

    model = LinearRegression()
    model.fit(X_feature.reshape(-1,1), y_train)

    coef = model.coef_[0]
    intercept = model.intercept_

    plt.scatter(X_feature, y_train, alpha=0.5, color='b')
    plt.plot(X_feature, coef * X_feature + intercept, color='r')

    plt.xlabel(feature_name)
    plt.ylabel("Target")
    plt.show()

    return coef, intercept
```

---

# 🧠 Pourquoi sklearn exige reshape(-1,1) ?

Parce que sklearn attend une matrice 2D :

```
(n_samples, n_features)
```

Même si tu as 1 feature.

Donc :

* `(1067,)` ❌
* `(1067,1)` ✅

---

# 🔥 Résumé rapide de ce que tu viens d’apprendre

| Objet         | reshape possible ? |
| ------------- | ------------------ |
| pandas Series | ❌                  |
| numpy array   | ✅                  |

Donc la règle :

> Avant d’utiliser reshape → assure-toi d’être en numpy.

---

Franchement ?

Là tu ne fais plus des erreurs de débutant.

Tu es en train de comprendre la différence :

* pandas
* numpy
* sklearn

Et ça c’est une étape importante 👌

Si tu veux, je peux te donner une règle mentale simple pour ne plus jamais te tromper entre DataFrame / Series / array.




"""