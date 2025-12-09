import pandas as pd
import numpy as np
df = pd.read_csv("Data_Supbio.csv")
df = df.apply(pd.to_numeric, errors='ignore')

print(df.columns)
print(df.shape[0])
liste_quantitative = []

for columns in df.columns:
    quantitative = False
    for x in df[columns]:
            if x is float or x is int:
                quantitative = True
    print(quantitative)
    if quantitative:liste_quantitative.append(columns)

print(liste_quantitative)
            


