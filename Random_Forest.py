import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import  r2_score, root_mean_squared_error
import numpy as np
data = pd.read_parquet("cleaned_data.parquet")

liste_spé =[
    'Biologie/Ecologie',
    'Mathématiques',
    'Physique-Chimie',
    "Sciences_de_l'ingénieur",
    "Sciences_de_l'ingénieur_et_sciences_physiques",
    'Sciences_de_la_vie_et_de_la_Terre',
    "Langues,_littératures_et_cultures_étrangères_et_régionales",
    'Sciences_économiques_et_sociales',
    'Humanités,_littérature_et_philosophie',
    'Numérique_et_sciences_informatiques',
    "Histoire-Géographie,_Géopolitique_et_Science_politiques"
    
]

for spé in liste_spé:
    data[spé] = (
        (data["Spé_A"] == spé) |
        (data["Spé_B"] == spé) |
        (data["Spé_C"] == spé)
    ).astype(int)


data = data.drop(columns=["Spé_A", "Spé_B", "Spé_C","PSpé_A", "PSpé_B"])
data = data.replace({"Oui": 1, "Non": 0, "GC": 25})
data.loc[data["TT1Maths_Expertes"].isna(), "TT1Maths_Expertes"] = 0
data.loc[data["TT1Maths_Expertes"].notna(), "TT1Maths_Expertes"] = 1
X = data.drop(columns=["Note_Profil_Ecole"])
y = data["Note_Profil_Ecole"]

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Split (mets random_state pour reproductibilité)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(random_state=42, n_jobs=-1)

param_dist = {
    "n_estimators": [200, 400, 800, 1200],
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 3, 5, 10],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False],
}

search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=40,                  # augmente si tu veux
    scoring="neg_root_mean_squared_error",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=2
)

search.fit(X_train, y_train)

print("Meilleurs paramètres :", search.best_params_)
print("Meilleur CV RMSE :", -search.best_score_)

best_model = search.best_estimator_
y_pred = best_model.predict(X_test)


# Scores
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Test RMSE :", rmse)
print("Test R²   :", r2)

# Ecart-types (comparaison utile)
y_test_num = pd.to_numeric(y_test, errors="raise")      # convertit y_test (pas y_pred)
print("y_test std :", y_test_num.std())
print("y_pred std :", np.std(y_pred))                   # y_pred est un numpy array

# Feature importances
importances = best_model.feature_importances_
feat_imp = pd.DataFrame({
    "feature": X_train.columns,
    "importance": importances
}).sort_values("importance", ascending=False)

print(feat_imp.head(15))
