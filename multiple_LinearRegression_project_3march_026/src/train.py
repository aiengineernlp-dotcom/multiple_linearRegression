import joblib
import numpy as np
import pandas as pd
import shape

from config import *
#load data
df = pd.read_csv(RAW_DATA_FILE)
print(f"Affichage des premieres lignes du dataset\n: {df.head().to_string()}")
print(f"affichage de quelques lignes aleatoires du dataset\n {df.sample(9)}")

# exploration et coomprehension des donnees
print(f"description des donnees\n {df.describe().to_string()}")
print(df.columns)
print(df.info())

''' SECTION DE VARIABLES '''

# je vais retirer les colonnes/variables que je ne vais pas utiliser ici pour ce cas
df = df.drop(['MODELYEAR','MAKE','MODEL','VEHICLECLASS','TRANSMISSION','FUELTYPE',],axis=1) # on les retire car ce sont des chaines de caracteres
# on regarde maintenant la relation de correlation quil ya avec les variables restantes puis on choisit les meilleurs
# (ceux qui sont plus correllee avec la variable cible) et si deux sont fortement correllees entre elle je choisie celle qui est plus correlle avec la cible
#print(f"correlation entre les features\n {df.corr().to_string()}")   #Evite de chauffer la machine
#selection des features
 ##- je remarque que ENGINESIZE et CYLINDERS sont fortement correller entre elle mais juste ENGINESIZE est plus correlle avec la cible donc je prend ENGINESIZE
 ##- je prend aussi FUELCONSUMPTION_COMB_MPG car plus correlle avec la cible
df = df.drop(['CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB'],axis=1)
# print(df.corr())  #Evite de chauffer la machine


X = df[FEATURES_TO_USE_3] #2D
y = df[TARGET_VARIABLE] #1D

# verification des shapes
print(f"OBSERVATION DES VARIABLES EXPLICATIVES\n",X)
print(X.shape)
print(f"OBSERVATION DE la VARIABLES cible\n",y)
print(y.shape)

# separtaion de des donnees en test et train set pour eviter le data leakage
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y, test_size = TEST_SIZE, random_state = RANDOM_STATE)
# vu que les donnees de FUELCONSUMPTION_COMB_MPG sont trop grand par rapport au donnes de ENGINESIZE , je vais faire le scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_transform =  scaler.transform(X_test)

#note que a cause de transform, X_test n'est plus un dataframe mais un numpy array  donc en tenir compte pour la suite
#creation du model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# les coeficients
coef = model.coef_
intercept = model.intercept_

print(f"coefficient : {coef}")
print(f"interceipt : {intercept}")

# prediction
y_pred = model.predict(X_test_transform)
print(f"Prediction {y_pred}")

# Get the standard scaler's mean and standard deviation parameters
mean_=  scaler.mean_
std_=  scaler.scale_
print(f"mean du scaler : {mean_}")
print(f"std du scaler : {std_}")

# The least squares parameters can be calculated relative to the original, unstandardized feature space as:

coef_original= coef/std_
intercept_original = intercept - np.sum(mean_*coef/std_)
print(f"coefficient original : {coef_original}")
print(f"intercept original : {intercept_original}")

#save le model
joblib.dump(model,MODEL_FILE)

#save les tests set
joblib.dump(X_test,X_TEST_FILE)
joblib.dump(y_test,Y_TEST_FILE)
joblib.dump(X_test_transform,X_TEST_TRANSFORM)
joblib.dump(X_train_scaled,X_TRAIN_SCALED)
joblib.dump(X_train,X_TRAIN_FILE)
joblib.dump(y_train,Y_TRAIN_FILE)
#sauvegarder le scatter
joblib.dump(scaler,SCALES_FILE)

#sauvegarder le database_que je vais utiliser
joblib.dump(df,FINAL_DF)

print("✅ Modèle entraîné et sauvegardé.")
print("✅ Test set sauvegardé.")

print("\n")
"""CETTE PARTIE EST POUR LA VISUALISATION"""
df_feature_index = df.columns.get_loc(FEATURES_TO_USE_1[0]) #
print(df_feature_index) # me renvoie 0 donc ma colonne 1

"""
🎯 Ligne complète
df.columns.get_loc(FEATURES_TO_USE_2[0])
Supposons :
FEATURES_TO_USE_2 = ['ENGINESIZE']

🧠 Étape 1 — FEATURES_TO_USE_2[0]
FEATURES_TO_USE_2 est une liste.
Donc :
FEATURES_TO_USE_2[0]
👉 retourne la string :
'ENGINESIZE'

🧠 Étape 2 — df.columns
Si ton dataframe est :
ENGINESIZE | CYLINDERS | FUELCONSUMPTION
Alors :
df.columns
retourne un objet Index :
Index(['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION'], dtype='object')
C’est juste la liste des noms de colonnes.

🧠 Étape 3 — .get_loc()
get_loc() veut dire :
“Donne-moi la position (location) de cette colonne”
Donc :
df.columns.get_loc('ENGINESIZE')
retournera :
0
Parce que ENGINESIZE est la première colonne.
Si tu fais :
df.columns.get_loc('CYLINDERS')
Ça retourne :
1
🎯 Donc la ligne complète veut dire :
👉 "Trouve-moi la position numérique de la colonne 'ENGINESIZE' dans le dataframe"
"""








