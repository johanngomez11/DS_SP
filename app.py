import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Cargar el modelo previamente entrenado
model = joblib.load("best_wine_model.pkl")

# Definir los nombres de las columnas del dataset
column_names = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280_od315_of_diluted_wines", "proline"
]

# Crear un pipeline con escalado estándar
pipeline = StandardScaler()

# Título de la aplicación
st.title("Clasificación de Vinos")
st.write("Ingrese los datos del vino para predecir su clase.")

# Función para cargar datos desde un formulario
def input_features():
    data = {}
    for col in column_names:
        # Permitir al usuario ingresar valores numéricos para cada característica
        data[col] = st.number_input(f"Ingrese {col}", value=0.0)
    return pd.DataFrame([data])

# Obtener los datos ingresados por el usuario
user_input = input_features()

# Botón para realizar la predicción
if st.button("Predecir"):
    # Escalar los datos usando el mismo pipeline que se usó durante el entrenamiento
    # Nota: Aquí no necesitas ajustar el scaler nuevamente porque ya fue ajustado durante el entrenamiento.
    user_input_scaled = pipeline.fit_transform(user_input)

    # Realizar la predicción
    prediction = model.predict(user_input_scaled)

    # Mostrar la predicción
    st.success(f"La clase predicha es: {prediction[0]}")