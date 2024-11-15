import streamlit as st
import pickle
import pandas as pd

with open('log_reg.pkl', 'rb') as lo:
    log_reg = pickle.load(lo)

with open('svc_m.pkl', 'rb') as sv:
    svc_m = pickle.load(sv)

with open('tree_clf.pkl', 'rb') as tr:
    tree_clf = pickle.load(tr)

with open('label_encoder.pkl', 'rb') as le_file:
    le = pickle.load(le_file)

def classify(num):
    return le.inverse_transform([num])[0]

def main():
    st.title('Clasificación de Especies de Peces')

    st.sidebar.header('Parámetros de Entrada del Usuario')

    def user_input_parameters():
        length = st.sidebar.slider('Length (cm)', float(5), float(40), float(10))
        weight = st.sidebar.slider('Weight (g)', float(1), float(1500), float(300))
        w_l_ratio = weight / length
        data = {'length': length, 'weight': weight, 'w_l_ratio': w_l_ratio}
        features = pd.DataFrame(data, index=[0])
        return features

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
