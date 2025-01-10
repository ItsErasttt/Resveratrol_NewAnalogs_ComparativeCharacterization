import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Загрузка данных
data = pd.DataFrame({
    'SMILES': ['C(=CC1=CC(=CC(=C1)O)O)C2=CC=C(C=C2)O',  # Ресвератрол
               'C1=CC(=NC=C1O)/C=C/C2=CC=C(C=C2)O',  # Аналог 1
               'C1=CC=CC(=C1)C2=NN=C(C3=CC=NC=C3)C=N2'],  # Аналог 2
    'Name': ['Resveratrol', 'Analog_1', 'Analog_2']
})

# 2. Проверка валидности SMILES
data['Valid'] = data['SMILES'].apply(lambda smi: Chem.MolFromSmiles(smi) is not None)
if not data['Valid'].all():
    print("Некорректные SMILES:")
    print(data[~data['Valid']])
    data = data[data['Valid']]

# 3. Функция для расчёта дескрипторов
def calculate_descriptors(smiles_list):
    descriptors = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            desc = [
                Descriptors.MolWt(mol),        # Молекулярная масса
                Descriptors.TPSA(mol),        # Полярная поверхность
                Descriptors.MolLogP(mol),     # Коэффициент гидрофобности
                Descriptors.NumHDonors(mol),  # Число доноров водородной связи
                Descriptors.NumHAcceptors(mol)  # Число акцепторов водородной связи
            ]
            descriptors.append(desc)
        else:
            descriptors.append([np.nan] * 5)  # Если ошибка, то NaN
    return pd.DataFrame(descriptors, columns=['MolWt', 'TPSA', 'LogP', 'HDonors', 'HAcceptors'])

# Расчёт дескрипторов
descriptors = calculate_descriptors(data['SMILES'])
data = pd.concat([data, descriptors], axis=1)

# 4. Удаление NaN
data.dropna(inplace=True)

# 5. Нормализация дескрипторов
scaler = StandardScaler()
scaled_descriptors = scaler.fit_transform(data[['MolWt', 'TPSA', 'LogP', 'HDonors', 'HAcceptors']])

# 6. Кластеризация
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_descriptors)

# 7. Визуализация кластеров
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=data['MolWt'], y=data['LogP'], hue=data['Cluster'], palette='viridis', s=100
)
plt.title('Кластеризация аналогов ресвератрола')
plt.xlabel('Молекулярная масса')
plt.ylabel('LogP')
plt.legend(title='Кластер')
plt.show()

# 8. Вывод данных
print("Аналоги ресвератрола с кластерами:")
print(data[['Name', 'SMILES', 'Cluster']])

# Сохранение данных
data.to_csv("resveratrol_analogs_clusters.csv", index=False)
