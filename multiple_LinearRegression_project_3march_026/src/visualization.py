#visualisation
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import seaborn as sns
from config import *

import joblib
model = joblib.load(MODEL_FILE)
y_test = joblib.load(Y_TEST_FILE)
X_test_transform = joblib.load(X_TEST_TRANSFORM)
print(X_test_transform.shape) # avec le shape j'ai (214, 2) et avec le print(X_test_transform.shape[1]) j'obtiens 2
from src.train import intercept

#load model pour la fonction de multiple features
X_Train_scaled = joblib.load(X_TRAIN_SCALED)
#load Final DF
df_final = joblib.load(FINAL_DF)

df_feature_index_simple = df_final.columns.get_loc(FEATURES_TO_USE_1[0]) #
df_feature_index_multiple_1 = df_final.columns.get_loc(FEATURES_TO_USE_1[0]) #
df_feature_index_multiple_2 = df_final.columns.get_loc(FEATURES_TO_USE_2[0]) #
# ceci est pour le code de -->> plot_regression_plane_3d()
# df_feature_index = df_final.columns.get_loc(FEATURES_TO_USE_5[0])#ne peut pas fonctionner car n'est pas dans df_final
# print(df_feature_index_simple) # me renvoie 0 donc ma colonne 1
# print(FEATURES_TO_USE_1[0]) # me renvoie ENGINESIZE


def plot_predicted_vs_actuals_scatter(y_true, y_pred):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")

    # ligne parfaite y = x
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val])

    plt.title("Predicted vs Actual")
    plt.show()

# 👉 Ça visualise la qualité du modèle.
# 👉 Plus les points sont proches de la diagonale → meilleur modèle.

def plot_simple_regression_feature(X_train,y_train,df_feature_index_simple,feature_name):
    """
    Fit and plot simple linear regression for one feature.
    NB: il ne s'agit pas de mon modele final qui se trouve dans mon train.py. Celui la est meme scaler et ici non
    """
    X_feature = X_train.iloc[:,df_feature_index_simple].values # pourquoi on met.values? comprend ca ici: src/evaluate.py en bas
    # aussi rappele toi que on veut avoir X_feature en 2d or ici il ya pas de moyen direct pour faire double crochets
    # alors on converti dataframe to numpy.
    '''
      # note que cette syntaxe: "X_train[:, df_feature_index_simple]" fonctionne uniquement avec numpy arrays.Pas avec pandas
      j'ai passe trop de temps ici a cause d'une petite erreur liee a cette syntaxe ci-haut. or 
      En pandas, on ne sélectionne PAS comme ça.par contre, on fait :
      Pour un DataFrame, on utilise :
        .iloc → position
        .loc → nom de colonne
        
        et non "X_train[:, df_feature_index]" sans les .iloc (position) ou .loc(nom de colonne)
        
        -------------------------------
        
        🧠 Ce que tu viens d’apprendre
            DataFrame ≠ numpy array.
            [:, index] → numpy
            .iloc[:, index] → pandas
    '''

    model = LinearRegression()
    model.fit(X_feature.reshape(-1,1),y_train)
    coef = model.coef_[0] # on extrait l'unique valeur CAR SANS [0] ON A: array([15.32]) ET AVEC [0] ON A
    intercept = model.intercept_ # déjà une valeur simple Exemple : 120.5

    '''
    coef = model.coef_[0] # on extrait l'unique valeur CAR SANS [0] ON A: array([15.32]) ET AVEC [0] ON A 
    intercept = model.intercept_ # déjà une valeur simple Exemple : 120.5
    🎯 Ce qui se passe vraiment
    Quand tu fais : model.fit(X_feature.reshape(-1,1), y_train)
    Tu as :
    1 feature
    1 target
    Donc sklearn crée :  model.coef_  → array([valeur]); model.intercept_ → float
    Même en régression simple, coef_ reste un array.
    ----------------------------------------------------------------------------------------------
    Pourquoi ?
    Parce que sklearn est conçu pour fonctionner de la même manière en :
    régression simple
    régression multiple
    multi-output
    C’est une question de cohérence interne.
    ----------------------------------------------------------------------------------------------
Comparaison simple vs multiple: 
🔹 Régression simple 
            - model.coef_      → array([15.2])
            - model.intercept_ → 120.5
    Donc on fait : coef = model.coef_[0]
🔹 Régression multiple (2 features)
    - model.coef_ → array([15.2, -3.8]) , Là [0] serait juste le coefficient de la première variable.
    
🎯 Donc la vraie réponse : Oui, tu vois [0] parce qu’il n’y a qu’une seule feature…
    '''
    plt.figure(figsize=(8, 8))
    plt.scatter(X_feature, y_train, alpha=0.5,color='b',label='distribution de X_feature par rapport a y_train')
    plt.plot(X_feature,coef*X_feature + intercept,color='r',label='Regression line')
    plt.xlabel(feature_name)
    plt.ylabel(TARGET_VARIABLE)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return coef, intercept


def plot_multiple_model_projection(X_Train_scaled,y_train,model,feature_name,df_feature_index_multiple_1):
    """
    Plot projection of multiple regression model on one feature.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    feature_name = FEATURES_TO_USE_1[0]
    plt.figure(figsize=(10, 6))

    # Moyenne de toutes les features
    X_mean = np.mean(X_Train_scaled, axis=0)

    # Valeurs de la feature qu’on veut faire varier
    X_values = np.linspace(
        X_Train_scaled[:, df_feature_index_multiple_1].min(),
        X_Train_scaled[:, df_feature_index_multiple_1].max(),
        100
    )

    # On crée une matrice où toutes les features = moyenne
    X_temp = np.tile(X_mean, (100, 1))

    # On remplace uniquement la feature choisie
    X_temp[:, df_feature_index_multiple_1] = X_values

    # Prédictions avec le vrai modèle
    y_pred = model.predict(X_temp)

    # Scatter des vraies données
    plt.scatter(X_Train_scaled[:, df_feature_index_multiple_1], y_train, alpha=0.4)

    # Ligne du vrai modèle multiple
    plt.plot(X_values, y_pred)

    plt.xlabel(feature_name + " (scaled)")
    plt.ylabel("Target")
    plt.title("Projection of Multiple Linear Regression")
    plt.show()


    """
    mais lorsque je vais l'afficher il va resemble a une simple linear regression
    💥 Oui.
    Et c’est totalement normal.
    
    Quand tu affiches la projection d’un modèle multiple sur une seule variable,
    ça va ressembler à une simple linear regression.
    
    Mais attention…
    
    👉 Ce n’est pas une régression simple.
    👉 C’est une projection d’un plan.
    🎯 Pourquoi ça ressemble à une droite ?
    Ton modèle multiple (avec 2 features) est : y=β1x1+β2x2+b ; C’est un plan en 3D.
    à une valeur constante (par exemple sa moyenne) : y=β1x1+(β2⋅constante)+b
    Ce qui devient :y=β1x1+C, Et ça… 👉 c’est une droite.
    🎯 Donc quand tu affiches : plt.plot(X_values, model.predict(X_temp))
    Tu affiches :
    👉 Une coupe du plan
    👉 Donc une droite
    👉 Donc ça ressemble à une simple regression
    Mais mathématiquement, ce n’est pas la même chose.
    
    🔥 Différence profonde
    | Régression simple               | Projection multiple            |
| ------------------------------- | ------------------------------ |
| Modèle entraîné avec 1 variable | Modèle entraîné avec plusieurs |
| Ignore les autres variables     | Les autres sont fixées         |
| Droite “réelle”                 | Droite “section du plan”       |


🎯 Pourquoi c’est important ?
Parce que :
Si tu fais : coef * X_feature + intercept
Tu ignores les autres variables.
Mais si tu fais : model.predict(X_temp)
Tu respectes le modèle complet.
    Visuellement → les deux ressemblent à une droite
    Mathématiquement → ce n’est pas la même chose.
    
    
==================================================================================================================
    
    
===========================>>>>>>>>>>>>c;est la raison on fait en 3d pour tout voir ?
💥 Exactement.
Oui — si tu as 2 variables, la visualisation 3D permet de voir le modèle complet.

Mais attention… ça a des limites.
🎯 Cas idéal : 2 features

Si ton modèle est : y=β1x1+β2x2+b

Alors tu peux faire :
    - Axe X → feature 1
    - Axe Y → feature 2
    - Axe Z → prédiction
Et là tu verras :
👉 un plan
Donc oui, en 3D tu vois le modèle entier.
---------------------------------------------------------------------------------------------------------

🧠 Pourquoi ça marche ?
Parce que :
1 feature → droite (2D)
2 features → plan (3D)
3 features → hyperplan (4D)
4 features → hyperplan (5D)
etc…
Et nous, on vit dans un monde 3D 😄
Donc au-delà de 2 features…
❌ Impossible de tout visualiser.
---------------------------------------------------------------------------------------------------------
🎯 Résumé clair

| Nombre de features | Forme géométrique | Visualisable ? |
| ------------------ | ----------------- | -------------- |
| 1                  | Droite            | ✅ Oui          |
| 2                  | Plan              | ✅ Oui (3D)     |
| 3                  | Hyperplan 4D      | ❌ Non          |
| 5                  | Hyperplan 6D      | ❌ Non          |


---------------------------------------------------------------------------------------------------------

🔥 Pourquoi on ne fait pas toujours du 3D ?
Même avec 2 features :
Graphiques 3D sont plus difficiles à lire
Interprétation moins intuitive
Peu pratique pour rapport / présentation
Donc en pratique :
👉 On utilise souvent des projections 2D
👉 Ou des partial dependence plots

🧠 Important
Le 3D ne change pas le modèle.
Ça change juste la manière dont tu le regardes.

🎯 Conclusion

Oui :
Si tu as exactement 2 features → 3D montre tout.
Mais si tu as plus de 2 → tu es obligé de projeter.
---------------------------------------------------------------------------------------------------------

    """


def plot_regression_plane_3d(X_test_transform,y_test,model):
    """
    3D visualization of multiple linear regression plane.
    Only works if X_test has exactly 2 features.
    """
    if X_test_transform.shape[1] != 2:
        raise ValueError('X_test must have only 2 features')

    #Extract features  ---------------->>>> ON N"OUBLIE PAS QUE Dans le train.py, après transform,
    # X_test devient un numpy array Donc ici :
    '''
    X1 = X_test_transform.iloc[:,0] # feature colonne 1
    X2 = X_test_transform.iloc[:,1] # feature colonne 2
    
    ❌ va planter si X_test_transform est un numpy array. car .iloc[] est pour les DataFrames
    Il faut plutôt faire sans iloc[]  :
'''
    X1 = X_test_transform[:, 0]
    X2 = X_test_transform[:, 1]

    #create a mesh grid
    x1_surf , x_2_surf = np.meshgrid(
        np.linspace(X1.min(), X1.max(), 100),
            np.linspace(X2.min(), X2.max(), 100)
    )


    # get model parameters
    intercept = model.intercept_
    slope = model.coef_

    # compute the regression plane

    y_surf = intercept + slope[0] * x1_surf + slope[1] * x_2_surf
    # predictions
    y_pred = model.predict(X_test_transform)

    # separation des points

    above_plane = y_test>=y_pred
    below_plane = y_test<y_pred

    #plot

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111,projection='3d')

    #    # Above / below plane
    ax.scatter(X1[above_plane],X2[above_plane],y_test[above_plane],s=70, alpha=.7, ec='k', label="Above Plane")
    ax.scatter(X1[below_plane],X2[below_plane],y_test[below_plane],s=50, alpha=.3, ec='k', label="Below Plane")
    ax.plot_surface(x1_surf, x_2_surf, y_surf, alpha=0.3)
    ax.view_init(elev=30, azim=45)
    ax.legend(loc='upper center')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Target')
    ax.set_title('Multiple Linear Regression Plane')

    plt.tight_layout()
    plt.show()



"""
Oui, exactement 👍 Tu touches au point clé.

Quand tu fais une **Multiple Linear Regression**, ton modèle ressemble à ça :

y=β0+β1x1+β2x2

Donc :

* **x₁ = Feature 1**
* **x₂ = Feature 2**
* **y = Target**

---

## 1️⃣ Pourquoi le graphique est en 3D ?

Parce que tu as **2 variables d'entrée + 1 sortie**.

Donc l'espace devient :

* Axe **X** → Feature 1
* Axe **Y** → Feature 2
* Axe **Z** → Target (y)

Le modèle n'est plus une **ligne**, mais un **plan**.

Visuellement :

* **Simple Linear Regression**

y=β0+β1x

➡️ ça donne **une ligne**

---

* **Multiple Linear Regression (2 features)**

y=β0+β1x1+β2x2

➡️ ça donne **un plan**

Ton image montre exactement ça.

---

## 2️⃣ Ce que représente ton graphique

Dans ton image :

* les **points** = tes données réelles
* le **plan bleu** = ce que le modèle prédit

Donc le modèle essaie de placer un **plan qui passe au plus près des points**.

Les points :

* **au-dessus du plan** → erreur positive
* **en dessous du plan** → erreur négative

C'est pour ça que tu vois :

* 🔵 Above Plane
* 🟠 Below Plane

---

## 3️⃣ Pourquoi parfois on montre une "simple regression"

Quand je disais :

> "visualiser l'effet d'une feature"

c'est une **astuce pédagogique**.

On **fixe une variable**.

Par exemple :

```
bedrooms = 3
```

Et on regarde seulement :

```
size → rent
```

Donc temporairement ça ressemble à :

y=β0+β1size

Mais en réalité le modèle reste :

y=β0+β1size+β2bedrooms

On **observe juste une tranche du plan**.

---

## 4️⃣ La vraie intuition

Imagine une **table inclinée**.

* la table = **le plan de régression**
* les billes = **les données**

Le modèle ajuste l'inclinaison de la table pour passer **au milieu des billes**.

---

## 5️⃣ Quand tu as plus de 2 features

Si tu as :

```
size
bedrooms
age
distance
crime_rate
```

le modèle devient :

y=β0+β1x1+β2x2+β3x3+...

Mais là :

❌ impossible de visualiser

car le plan devient un **hyperplane en 5D**.

Donc la 3D que tu as faite est **la meilleure visualisation possible**.

---

💡 Franchement Erman, ce que tu fais (créer tes propres datasets + visualiser) est **exactement la méthode des vrais ML engineers**.

Si tu veux, je peux aussi te montrer un truc **très puissant que 90% des débutants ne comprennent pas** :

👉 **comment voir les coefficients β comme l’inclinaison réelle du plan** (et pourquoi ton modèle retrouve 25, 8000, -1500).




"""



















