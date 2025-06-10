import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import implicit

# =========================
# Cargar y procesar datos
# =========================

@st.cache_data
def load_data():
    df = pd.read_csv("museos_cdmx.csv")
    df.columns = df.columns.str.encode('latin1').str.decode('utf-8')
    df['Centro de trabajo'] = df['Centro de trabajo'].str.encode('latin1').str.decode('utf-8')
    df['Nacionalidad'] = df['Nacionalidad'].str.encode('latin1').str.decode('utf-8')
    df['Tipo de visitantes'] = df['Tipo de visitantes'].str.encode('latin1').str.decode('utf-8')
    df['Segmento'] = df['Nacionalidad'] + " - " + df['Tipo de visitantes']
    return df

df = load_data()

# Crear índices
user_ids = {u: i for i, u in enumerate(df['Segmento'].unique())}
item_ids = {i: j for j, i in enumerate(df['Centro de trabajo'].unique())}
reverse_item_ids = {v: k for k, v in item_ids.items()}

df['user_index'] = df['Segmento'].map(user_ids)
df['item_index'] = df['Centro de trabajo'].map(item_ids)

# Matriz de interacciones
matrix = coo_matrix(
    (df['Número de visitas'], (df['user_index'], df['item_index']))
).tocsr()

# =========================
# Modelo ALS (sin cache)
# =========================

def train_model(matrix):  # sin @st.cache_resource
    model = implicit.als.AlternatingLeastSquares(
        factors=20, regularization=0.1, iterations=20
    )
    model.fit(matrix)
    return model

model = train_model(matrix)

# =========================
# Interfaz de usuario
# =========================

st.title("🎨 Recomendador de Museos en CDMX")
st.write("Este sistema recomienda museos con base en visitas históricas por segmento de visitantes.")

# Selección de segmento
segmento_seleccionado = st.selectbox("Selecciona tu segmento:", list(user_ids.keys()))
user_idx = user_ids[segmento_seleccionado]

# Botón para recomendar
if st.button("🔍 Ver recomendaciones"):
    st.session_state['recomendaciones'] = model.recommend(user_idx, matrix[user_idx], N=5)

if 'recomendaciones' in st.session_state:
    item_ids_recom, scores = st.session_state['recomendaciones']
    st.subheader("🎯 Museos recomendados:")
    for item_id, score in zip(item_ids_recom, scores):
        st.write(f"✅ {reverse_item_ids[item_id]} — Score: {score:.2f}")
    st.write("Estas recomendaciones se basan en el historial de visitas de tu segmento.")
# Información adicional
st.sidebar.header("ℹ️ Información adicional")
st.sidebar.write(
    "Este sistema utiliza un modelo de filtrado colaborativo para recomendar museos "
    "basado en el historial de visitas por segmento de visitantes. "
    "Los segmentos incluyen nacionalidades y tipos de visitantes."
)
# Información de contacto
st.sidebar.header("📞 Contacto")
st.sidebar.write(
    "Para más información, por favor contacta a:"
)
st.sidebar.write(
    "📧 Email: museandoiando@gmail.com"
)
st.sidebar.write(
    "📞 Teléfono: +52 55 5167 3208"
)
