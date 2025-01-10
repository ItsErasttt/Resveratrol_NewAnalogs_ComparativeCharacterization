import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, DataStructs, Draw, rdFMCS
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Входные данные
resveratrol_smiles = "C1=CC(=CC=C1/C=C/C2=CC(=C(C=C2)O)O)O"
analog_smiles = [
    "C1=CC(=CC=C1/C=C/C2=CC(=C(C=C2)O)O)",
    "C1=CC(=NC=C1O)/C=C/C2=CC=C(C=C2)O"
]

# Функция получения данных из PubChem через REST API
def fetch_pubchem_data(smiles):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/property/MolecularWeight,XLogP,TPSA,IUPACName/JSON"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        properties = response.json().get("PropertyTable", {}).get("Properties", [{}])[0]
        return {
            "Molecular Weight (PubChem)": properties.get("MolecularWeight"),
            "LogP (PubChem)": properties.get("XLogP"),
            "TPSA (PubChem)": properties.get("TPSA"),
            "IUPAC Name": properties.get("IUPACName"),
        }
    except Exception as e:
        print(f"Ошибка получения данных из PubChem для {smiles}: {e}")
    return {}

# Функция расчета локальных дескрипторов
def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return {
            "SMILES": smiles,
            "Molecular Weight": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "Num H-Donors": Descriptors.NumHDonors(mol),
            "Num H-Acceptors": Descriptors.NumHAcceptors(mol)
        }
    return None

# Получение данных
data = []
for sm in [resveratrol_smiles] + analog_smiles:
    local_descriptors = calculate_descriptors(sm)
    pubchem_data = fetch_pubchem_data(sm)
    combined_data = {**local_descriptors, **pubchem_data}
    data.append(combined_data)

# Создание сравнительной таблицы
df = pd.DataFrame(data)
print("\nСравнительная таблица:")
print(df)

# Оценка сходства через коэффициент Танимото
def calculate_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    fp1 = GetMorganFingerprintAsBitVect(mol1, radius=2)
    fp2 = GetMorganFingerprintAsBitVect(mol2, radius=2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

df["Similarity with Resveratrol"] = [1.0] + [calculate_similarity(resveratrol_smiles, sm) for sm in analog_smiles]

# Генерация новых молекул на основе MCS
print("\nГенерация новых молекул на основе MCS:")
base_mol = Chem.MolFromSmiles(resveratrol_smiles)
analog_mols = [Chem.MolFromSmiles(sm) for sm in analog_smiles]
mcs = rdFMCS.FindMCS(analog_mols + [base_mol], completeRingsOnly=True).smartsString
mcs_mol = Chem.MolFromSmarts(mcs)

# Визуализация молекул
mols = [base_mol] + analog_mols + [mcs_mol]
legends = ["Resveratrol"] + [f"Analog {i+1}" for i in range(len(analog_smiles))] + ["MCS"]
img = Draw.MolsToGridImage(mols, legends=legends)
img.show()

# Фармакологическая оценка
print("\nФармакологическая оценка:")
activity_data = {
    "SMILES": [resveratrol_smiles] + analog_smiles,
    "Activity": [1, 0, 1]  # Пример активности: 1 = активно, 0 = неактивно
}
activity_df = pd.DataFrame(activity_data)
descriptors_df = pd.merge(df, activity_df, on="SMILES")

# Подготовка данных
X = descriptors_df[["Molecular Weight", "LogP", "TPSA", "Num H-Donors", "Num H-Acceptors"]]
y = descriptors_df["Activity"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Прогноз и оценка точности
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nТочность модели: {accuracy:.2f}")

# Важность дескрипторов
importances = model.feature_importances_
plt.bar(X.columns, importances)
plt.title("Важность дескрипторов")
plt.ylabel("Вклад")
plt.xticks(rotation=45)
plt.show()

# Кластеризация молекул
print("\nКластеризация молекул:")
fp_list = [GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(sm), radius=2) for sm in [resveratrol_smiles] + analog_smiles]
fp_array = np.array([list(fp) for fp in fp_list])
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(fp_array)
df["Cluster"] = clusters
print(df[["SMILES", "Cluster"]])
