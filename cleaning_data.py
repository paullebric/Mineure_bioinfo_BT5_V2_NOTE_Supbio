import pandas as pd
import numpy as np
df = pd.read_csv("Data_Supbio.csv")
df = df.apply(pd.to_numeric, errors='ignore')

colonnes_voulues = [
    "ID_SF",
    "Note_bac",          # corrigé
    "Moyenne_Bac",
    "Pmoyenne",          # corrigé
    "Spé_A",
    "Spé_B",
    "Spé_C",
    "TT2Anglais",
    "TT1Anglais",
    "PSpé_A",
    "PSpé_B",
    "PMaths_Compl",
    "Note_Profil_Ecole"
]
df_clean = df[colonnes_voulues].copy()
df_clean = df_clean.set_index("ID_SF")

df_clean = df_clean.dropna()

df_clean.isna().sum().sum()

df_clean.to_parquet("cleaned_data.parquet")
