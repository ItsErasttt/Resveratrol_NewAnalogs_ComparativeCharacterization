import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Заданные молекулы в формате SMILES
molecules = {
    'Resveratrol': 'C(=CC1=CC(=CC(=C1)O)O)C2=CC=C(C=C2)O',  # SMILES-формула резвератрола
    'Analog_1': 'C1=CC(=NC=C1O)/C=C/C2=CC=C(C=C2)O',         # SMILES для Аналога 1
    'Analog_2': 'C1=CC=CC(=C1)C2=NN=C(C3=CC=NC=C3)C=N2',     # SMILES для Аналога 2
}

# Функция для получения молекулярных дескрипторов
def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    descriptors = {}
    descriptors['MolWeight'] = Descriptors.MolWt(mol)
    descriptors['NumHDonors'] = Descriptors.NumHDonors(mol)
    descriptors['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
    descriptors['TPSA'] = Descriptors.TPSA(mol)
    descriptors['LogP'] = Descriptors.MolLogP(mol)
    return descriptors

# Получаем дескрипторы для заданных молекул
data = []
for name, smiles in molecules.items():
    descriptors = get_descriptors(smiles)
    if descriptors:
        descriptors['SMILES'] = smiles
        data.append(descriptors)

# Преобразуем в DataFrame
df = pd.DataFrame(data)

# Визуализируем дескрипторы
df.set_index('SMILES')[['MolWeight', 'LogP', 'TPSA']].plot(kind='bar', figsize=(10, 6))
plt.title('Molecular Descriptors for Resveratrol and Analogues')
plt.ylabel('Descriptor Value')
plt.show()

# Подготовка данных для обучения
X = df[['MolWeight', 'LogP', 'TPSA']]
y = np.array([1.0, 2.0, 3.0])  # Примерные метки для молекул (можно заменить на реальные данные)

# Разделение на обучающие и тестовые данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Оценка модели
y_pred = model.predict(X_test)
print("MAE: ", mean_absolute_error(y_test, y_pred))

# Генерация новых молекул
def generate_new_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Пример модификации: добавление метильной группы (-CH3)
    editable_mol = Chem.RWMol(mol)
    editable_mol.AddAtom(Chem.Atom('C'))  # Добавляем атом углерода
    editable_mol.AddBond(0, editable_mol.GetNumAtoms() - 1, Chem.BondType.SINGLE)  # Связываем с первым атомом
    return editable_mol.GetMol()

# Применение генерации и визуализации
generated_molecules = []
for name, smiles in molecules.items():
    new_mol = generate_new_molecule(smiles)
    if new_mol:
        generated_molecules.append(new_mol)
        img = Draw.MolToImage(new_mol, size=(300, 300))
        img.show()

# Вывод новых молекул и их дескрипторов
for i, mol in enumerate(generated_molecules, 1):
    smiles = Chem.MolToSmiles(mol)
    descriptors = get_descriptors(smiles)
    print(f'Generated Molecule {i}:')
    print(f'SMILES: {smiles}')
    print(f'Descriptors: {descriptors}')
