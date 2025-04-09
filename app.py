import streamlit as st
import pandas as pd
import joblib

# Cargar el pipeline previamente entrenado
pipeline = joblib.load("best_wine_model.pkl")

# Definir los nombres de las columnas del dataset
column_names = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280_od315_of_diluted_wines", "proline"
]

# Mapeo de las clases a descripciones más amigables
class_descriptions = {
    1: "Vinos producidos con la primera variedad de uva.",
    2: "Vinos producidos con la segunda variedad de uva.",
    3: "Vinos producidos con la tercera variedad de uva."
}

# Título de la aplicación
st.title("Clasificación de Vinos")
st.write("Ingrese los datos del vino para predecir su clase.")

# Función para cargar datos desde un formulario
def input_features():
    data = {}
    for col in column_names:
        data[col] = st.number_input(f"Ingrese {col}", value=0.0)
    return pd.DataFrame([data])

# Obtener los datos ingresados por el usuario
user_input = input_features()

# Botón para realizar la predicción
if st.button("Predecir"):
    try:
        # Realizar la predicción usando el pipeline completo
        prediction = pipeline.predict(user_input)[0]

        # Mostrar la predicción y la descripción correspondiente
        st.success(f"La clase predicha es: {prediction}")
        st.info(f"Descripción: {class_descriptions[prediction]}")
    except Exception as e:
        st.error(f"Ocurrió un error: {str(e)}")
