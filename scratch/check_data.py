import pandas as pd

# Cargar datos
cols = ['ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity',
        'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS',
        'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc',
        'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD',
        'Meth', 'Mush', 'Nicotine', 'Semer', 'VSA']

df = pd.read_csv('drug_consumption.data', header=None, names=cols)

# Sustancias objetivo
drugs = ['Cannabis', 'Nicotine', 'Ecstasy', 'Coke']

# Convertir CL0-CL6 a 0-6
def to_val(cl):
    return int(cl[2])

for d in drugs:
    df[d + '_val'] = df[d].apply(to_val)

# Determinar sustancia principal (la de mayor nivel)
# En caso de empate, idxmax toma la primera (Cannabis > Nicotine > Ecstasy > Coke)
df['Main'] = df[[d + '_val' for d in drugs]].idxmax(axis=1)

# Ver distribución
print("Distribución de la 'Sustancia Principal':")
print(df['Main'].value_counts())

# Ver cuántos son "No usuarios" de las 4 (todos en CL0 o CL1)
df['Max_Val'] = df[[d + '_val' for d in drugs]].max(axis=1)
no_users = len(df[df['Max_Val'] <= 1])
print(f"\nPersonas que NO consumen ninguna de las 4 (CL0 o CL1): {no_users}")
print(f"Total registros: {len(df)}")
