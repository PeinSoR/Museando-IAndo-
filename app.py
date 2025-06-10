import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import coo_matrix
import folium
from streamlit_folium import st_folium
from fpdf import FPDF
import base64

# =========================
# Cargar y procesar datos
# =========================

@st.cache_data
def load_data():
    df = pd.read_csv("museos_cdmx_geolocalizado.csv", encoding="latin1")
    df.columns = df.columns.str.strip()  # Limpia espacios extra en encabezados
    df['Segmento'] = df['Nacionalidad'] + " - " + df['Tipo de visitantes']
    return df

df = load_data()

user_ids = {u: i for i, u in enumerate(df['Segmento'].unique())}
item_ids = {i: j for j, i in enumerate(df['Centro de trabajo'].unique())}
reverse_item_ids = {v: k for k, v in item_ids.items()}

df['user_index'] = df['Segmento'].map(user_ids)
df['item_index'] = df['Centro de trabajo'].map(item_ids)

# =========================
# Modelo Deep Learning
# =========================

class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=20):
        super(RecommenderNet, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        self.fc1 = nn.Linear(embedding_size * 2, 64)
        self.fc2 = nn.Linear(64, 16)
        self.output = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, user_indices, item_indices):
        user_embedded = self.user_embedding(user_indices)
        item_embedded = self.item_embedding(item_indices)
        x = torch.cat([user_embedded, item_embedded], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.output(x)
        return out.squeeze()

@st.cache_resource(show_spinner=False)
def train_deep_model(df, epochs=10, lr=0.01):
    num_users = len(user_ids)
    num_items = len(item_ids)

    model = RecommenderNet(num_users, num_items)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Dataset tensors
    users = torch.LongTensor(df['user_index'].values)
    items = torch.LongTensor(df['item_index'].values)
    ratings = torch.FloatTensor(df['Visitas'].values)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(users, items)
        loss = criterion(outputs, ratings)
        loss.backward()
        optimizer.step()

    return model

model = train_deep_model(df)

# =========================
# Función para recomendar
# =========================

def get_recommendations(model, user_idx, N=5):
    model.eval()
    num_items = len(item_ids)
    user_tensor = torch.LongTensor([user_idx] * num_items)
    item_tensor = torch.LongTensor(list(range(num_items)))
    with torch.no_grad():
        scores = model(user_tensor, item_tensor).numpy()

    # Ordenar indices por score descendente
    top_indices = np.argsort(scores)[::-1][:N]
    top_scores = scores[top_indices]

    return top_indices, top_scores

# =========================
# Interfaz de usuario
# =========================

st.title("🎨 Recomendador de Museos en CDMX")

segmento_seleccionado = st.selectbox("Selecciona tu segmento:", list(user_ids.keys()))
user_idx = user_ids[segmento_seleccionado]

if st.button("🔍 Ver recomendaciones"):
    recomendaciones = get_recommendations(model, user_idx, N=5)
    st.session_state['recomendaciones'] = recomendaciones

if 'recomendaciones' in st.session_state:
    item_ids_recom, scores = st.session_state['recomendaciones']
    st.subheader("🎯 Museos recomendados:")

    opciones = [f"{reverse_item_ids[item_id]} — Score: {score:.2f}" for item_id, score in zip(item_ids_recom, scores)]
    museo_seleccionado = st.selectbox("Selecciona un museo para ver más detalles:", opciones)

    seleccionado_index = opciones.index(museo_seleccionado)
    item_id = item_ids_recom[seleccionado_index]
    nombre_museo = reverse_item_ids[item_id]

    st.markdown(f"### 🏛️ {nombre_museo}")
    st.markdown(f"**Puntaje estimado:** {scores[seleccionado_index]:.2f}")

    detalles = df[df['Centro de trabajo'] == nombre_museo].drop_duplicates()
    st.write(detalles[['Periodo', 'Estado', 'Tipo de sitio', 'Tipo de visitantes', 'Nacionalidad', 'Visitas']])

    # =====================
    # MAPA con folium
    # =====================
    if not detalles[['Latitud', 'Longitud']].isnull().values.any():
        lat = detalles['Latitud'].values[0]
        lon = detalles['Longitud'].values[0]

        st.subheader("📍 Ubicación del museo")
        m = folium.Map(location=[lat, lon], zoom_start=15)
        folium.Marker([lat, lon], popup=nombre_museo).add_to(m)
        st_folium(m, width=700)

    # =====================
    # Exportar a PDF
    # =====================
    st.subheader("📄 Exportar recomendación")

    if st.button("📥 Descargar como PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Recomendación de Museo", ln=1, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, f"Museo recomendado: {nombre_museo}")
        pdf.multi_cell(0, 10, f"Puntaje: {scores[seleccionado_index]:.2f}")
        pdf.multi_cell(0, 10, f"Segmento: {segmento_seleccionado}")
        for col in ['Periodo', 'Estado', 'Tipo de sitio', 'Tipo de visitantes', 'Nacionalidad', 'Visitas']:
            valor = detalles[col].values[0]
            pdf.multi_cell(0, 10, f"{col}: {valor}")

        pdf_file = f"recomendacion_{nombre_museo.replace(' ', '_')}.pdf"
        pdf.output(pdf_file)

        with open(pdf_file, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="{pdf_file}">📥 Haz clic aquí para descargar el PDF</a>'
            st.markdown(href, unsafe_allow_html=True)

# =========================
# Información adicional
# =========================

st.sidebar.header("ℹ️ Información adicional")
st.sidebar.write(
    "Este sistema utiliza un modelo de filtrado colaborativo para recomendar museos "
    "basado en el historial de visitas por segmento de visitantes. "
    "Los segmentos incluyen nacionalidades y tipos de visitantes."
)

st.sidebar.header("📞 Contacto")
st.sidebar.write("📧 Email: museandoiando@gmail.com")
st.sidebar.write("📞 Teléfono: +52 55 5167 3208")
