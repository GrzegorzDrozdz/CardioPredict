import pandas as pd

# Wczytanie danych
df = pd.read_csv("heart.csv")

# Filtrowanie tylko osób bez choroby serca (i kopiowanie danych!)
df_no_disease = df[df["HeartDisease"] == 0].copy()

# Cechy ciągłe
cont_cols = ["RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

# Przedziały wiekowe: 28–34 (7 lat), potem co 5, do 77
age_bins = [28, 35, 40, 45, 50, 55, 60, 65, 70, 78]
age_labels = [f"{age_bins[i]}-{age_bins[i+1]-1}" for i in range(len(age_bins) - 1)]

# Dodanie kolumny z grupą wiekową (bez ostrzeżeń)
df_no_disease.loc[:, "AgeGroup"] = pd.cut(df_no_disease["Age"], bins=age_bins, labels=age_labels, right=False)

# Grupowanie i liczenie median z jawnie ustawionym observed=False
median_by_age = df_no_disease.groupby("AgeGroup", observed=False)[cont_cols].median().reset_index()

# Wyświetlenie wyników
print("Mediany cech ciągłych w grupach wiekowych (tylko osoby bez choroby serca):")
print(median_by_age)
