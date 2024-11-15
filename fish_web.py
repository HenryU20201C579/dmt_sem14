# importar librerías
import streamlit as st
import pickle
import pandas as pd
import os

# función para cargar modelos de forma segura
def load_pickle(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    else:
        st.error(f"Error: el archivo {file_path} no se encuentra.")
        return None

# cargar modelos y codificador de etiquetas
log_reg = load_pickle('log_reg.pkl')
svc_m = load_pickle('svc_m.pkl')
tree_clf = load_pickle('tree_clf.pkl')
le = load_pickle('label_encoder.pkl')

# verificar si los modelos y el codificador se cargaron correctamente
if log_reg is None or svc_m is None or tree_clf is None or le is None:
    st.stop()  # detener ejecución si hay archivos faltantes

# función para clasificar las especies
def classify(num):
    return le.inverse_transform([num])[0]

# función principal de la app
def main():
    st.title('Clasificación de Especies de Peces')
    st.sidebar.header('Parámetros de Entrada del Usuario')

    # función para capturar los parámetros del usuario
    def user_input_parameters():
        length = st.sidebar.slider('Length (cm)', 5.0, 40.0, 10.0)
        weight = st.sidebar.slider('Weight (g)', 1.0, 1500.0, 300.0)
        w_l_ratio = weight / length
        data = {'length': length, 'weight': weight, 'w_l_ratio': w_l_ratio}
        return pd.DataFrame(data, index=[0])

    df = user_input_parameters()
    option = ['Logistic Regression', 'SVM', 'Decision Tree']
    model = st.sidebar.selectbox('Selecciona el modelo que deseas usar:', option)
    st.subheader('Parámetros de Entrada del Usuario')
    st.write(df)

    if st.button('RUN'):
        if model == 'Logistic Regression':
            prediction = log_reg.predict(df)
        elif model == 'SVM':
            prediction = svc_m.predict(df)
        else:
            prediction = tree_clf.predict(df)
        
        st.success(f'La especie clasificada es: {classify(prediction[0])}')

if __name__ == '__main__':
    main()
