# importar librerías necesarias
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle

df = pd.read_csv("fish_data.csv")

le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

X = df[['length', 'weight', 'w_l_ratio']]
y = df['species']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

log_reg = LogisticRegression(max_iter=200)
svc_m = SVC()
tree_clf = DecisionTreeClassifier()

log_regr = log_reg.fit(x_train, y_train)
svc_mo = svc_m.fit(x_train, y_train)
tree_clf_mo = tree_clf.fit(x_train, y_train)

with open('log_reg.pkl', 'wb') as lo:
    pickle.dump(log_regr, lo)

with open('svc_m.pkl', 'wb') as sv:
    pickle.dump(svc_mo, sv)

with open('tree_clf.pkl', 'wb') as tr:
    pickle.dump(tree_clf_mo, tr)

with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(le, le_file)
