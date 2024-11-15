# importar librerías necesarias
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle

# cargar el dataset de especies de peces
df = pd.read_csv("fish_data.csv")

# convertir las etiquetas de especies a valores numéricos
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# seleccionar características y objetivo
X = df[['length', 'weight', 'w_l_ratio']]
y = df['species']

# dividir los datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# inicializar y entrenar modelos de clasificación
log_reg = LogisticRegression(max_iter=200)
svc_m = SVC()
tree_clf = DecisionTreeClassifier()

log_regr = log_reg.fit(x_train, y_train)
svc_mo = svc_m.fit(x_train, y_train)
tree_clf_mo = tree_clf.fit(x_train, y_train)

# guardar los modelos entrenados
with open('log_reg.pkl', 'wb') as lo:
    pickle.dump(log_regr, lo)

with open('svc_m.pkl', 'wb') as sv:
    pickle.dump(svc_mo, sv)

with open('tree_clf.pkl', 'wb') as tr:
    pickle.dump(tree_clf_mo, tr)

# guardar el codificador de etiquetas
with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(le, le_file)
