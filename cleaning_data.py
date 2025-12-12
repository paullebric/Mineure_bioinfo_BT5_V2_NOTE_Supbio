import pandas as pd
import numpy as np
df = pd.read_csv("Data_Supbio.csv")
df = df.apply(pd.to_numeric, errors='ignore')

colonnes_voulues = [
    "ID_SF",
    "Note_bac",          # corrigé
    "Moyenne_Bac",
    "Pmoyenne",
    "Tmoyenne",
    "Spé_A",
    "Spé_B",
    "Spé_C",
    "TT2Anglais",
    "TT1Anglais",
    "PT1Spé_A",
    "PT1Spé_B",
    "PT1Spé_C",
    "PT2Spé_A",
    "PT2Spé_B",
    "PT2Spé_C",
    "PSpé_A",
    "PSpé_B",
    "TT1Spé_A",
    "TT1Spé_B",
    "TT2Spé_A",
    "TT2Spé_B",
    "PMaths_Compl",
    "Note_Profil_Ecole",
    "TT1Maths_Expertes",
    "Bonus_Scientifique"
]
df_clean = df[colonnes_voulues].copy()
df_clean = df_clean.set_index("ID_SF")

df_clean = df_clean.dropna()

df_clean.isna().sum().sum()

df_clean.to_parquet("cleaned_data.parquet")
