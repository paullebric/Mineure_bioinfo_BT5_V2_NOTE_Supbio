import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import  r2_score, root_mean_squared_error
import numpy as np
data = pd.read_parquet("cleaned_data.parquet")

liste_spé =[
    'Arts_Plastiques',
    'Biologie/Ecologie',
    'Histoire-Géographie,_Géopolitique_et_Science_politiques',
    'Humanités,_Littérature_et_Philosophie',
    'Langues,_littératures_et_cultures_étrangères_et_régionales',
    'Mathématiques',
    'Musique',
    'Numérique_et_Sciences_Informatiques',
    'Physique-Chimie',
    "Sciences_de_l'ingénieur",
    "Sciences_de_l'ingénieur_et_sciences_physiques",
    'Sciences_de_la_vie_et_de_la_Terre',
    'Sciences_economiques_et_sociales'
]

for spé in liste_spé:
    data[spé] = (
        (data["Spé_A"] == spé) |
        (data["Spé_B"] == spé) |
        (data["Spé_C"] == spé)
    ).astype(int)


data = data.drop(columns=["Spé_A", "Spé_B", "Spé_C","PSpé_A", "PSpé_B"])
data = data.replace({"Oui": 1, "Non": 0, "GC": 25})

X = data.drop(columns=["Note_Profil_Ecole"])
y = data["Note_Profil_Ecole"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_reg = RandomForestRegressor(
    n_estimators=1000,
    max_depth=10,min_samples_split=5,min_samples_leaf=3,max_features='sqrt'
)

rf_reg.fit(X_train, y_train)

y_pred = rf_reg.predict(X_test)

print("RMSE :", root_mean_squared_error(y_test, y_pred))
print("R²   :", r2_score(y_test, y_pred))

y_true = y_test.astype(float)
y_hat = y_pred

plt.figure(figsize=(7, 7))

# Nuage de points
plt.scatter(y_true, y_hat, alpha=0.6)

# Diagonale idéale (prédictions parfaites)
mini = min(y_true.min(), y_hat.min())
maxi = max(y_true.max(), y_hat.max())
plt.plot([mini, maxi], [mini, maxi])

# Mise en forme
plt.xlabel("Observations")
plt.ylabel("Prédictions")
plt.title("Comparaison Observations vs Prédictions (Random Forest)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')

plt.show()