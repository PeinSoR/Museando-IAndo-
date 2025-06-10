import streamlit as st #Contruyendo la interfaz de usuario web
import pandas as pd # Manipulación de datos
import numpy as np # Operaciones numéricas
import torch # Biblioteca de PyTorch para Deep Learning
import torch.nn as nn # Módulo de redes neuronales de PyTorch
import torch.optim as optim # Optimizadores de PyTorch
from scipy.sparse import coo_matrix # Matrices dispersas para representar interacciones
import folium # Biblioteca para crear mapas interactivos
from streamlit_folium import st_folium # Integración de Folium con Streamlit
from fpdf import FPDF # Biblioteca para generar PDFs
import base64 # Para codificar archivos en base64

# =========================
# Cargar y procesar datos
# =========================

@st.cache_data # Caching de datos
def load_data(): # Carga y preprocesamiento de datos
    # Cargar el dataset de museos en CDMX
    df = pd.read_csv("museos_cdmx_geolocalizado.csv", encoding="utf-8")
    df.columns = df.columns.str.strip()  # Limpia espacios extra en encabezados
    df['Segmento'] = df['Nacionalidad'] + " - " + df['Tipo de visitantes'] # Crea una columna combinada de segmento
    return df

df = load_data() # Cargar los datos
st.title("🎨 Recomendador de Museos en CDMX") 
user_ids = {u: i for i, u in enumerate(df['Segmento'].unique())} # Mapeo de segmentos a índices
item_ids = {i: j for j, i in enumerate(df['Centro de trabajo'].unique())} # Mapeo de museos a índices
reverse_item_ids = {v: k for k, v in item_ids.items()} # Mapeo inverso de índices a museos

df['user_index'] = df['Segmento'].map(user_ids) # Mapea los segmentos a índices
df['item_index'] = df['Centro de trabajo'].map(item_ids) # Mapea los museos a índices

# =========================
# Modelo Deep Learning
# =========================

class RecommenderNet(nn.Module): # Definición de la red neuronal para recomendaciones
    def __init__(self, num_users, num_items, embedding_size=20): # Inicialización de la red
        super(RecommenderNet, self).__init__() # Llama al constructor de la clase base
        self.user_embedding = nn.Embedding(num_users, embedding_size) # Embedding para usuarios
        self.item_embedding = nn.Embedding(num_items, embedding_size) # Embedding para items (museos)
        self.fc1 = nn.Linear(embedding_size * 2, 64) # Capa densa para combinar embeddings
        self.fc2 = nn.Linear(64, 16) # Capa densa adicional
        self.output = nn.Linear(16, 1) # Capa de salida para predecir la puntuación
        self.relu = nn.ReLU() # Función de activación ReLU

    def forward(self, user_indices, item_indices): # Método de propagación hacia adelante
        #usar los embeddings de usuarios e items
        user_embedded = self.user_embedding(user_indices) 
        item_embedded = self.item_embedding(item_indices)
        x = torch.cat([user_embedded, item_embedded], dim=-1) # Concatenar los embeddings de usuario e item
        x = self.relu(self.fc1(x)) # Aplicar la primera capa densa y ReLU
        x = self.relu(self.fc2(x)) # Aplicar la segunda capa densa y ReLU
        out = self.output(x) # Capa de salida para obtener la puntuación final
        return out.squeeze() # Devuelve la puntuación como un vector unidimensional

@st.cache_resource(show_spinner=False) # Caching del modelo para evitar reentrenamiento innecesario
def train_deep_model(df, epochs=10, lr=0.01): # Entrenamiento del modelo de Deep Learning
    num_users = len(user_ids) # Número de usuarios (segmentos)
    num_items = len(item_ids) # Número de items (museos)
    # Inicializar el modelo, la función de pérdida y el optimizador

    model = RecommenderNet(num_users, num_items) # Crear una instancia del modelo
    criterion = nn.MSELoss() # Función de pérdida MSE para regresión
    optimizer = optim.Adam(model.parameters(), lr=lr) # Optimizador

    users = torch.LongTensor(df['user_index'].values) # Convertir los índices de usuarios a tensores
    items = torch.LongTensor(df['item_index'].values) # Convertir los índices de items a tensores
    ratings = torch.FloatTensor(df['Visitas'].values) # Convertir las visitas a tensores

    model.train() # Poner el modelo en modo de entrenamiento
    # Entrenamiento del modelo
    for epoch in range(epochs): 
        optimizer.zero_grad()
        outputs = model(users, items) 
        loss = criterion(outputs, ratings) 
        loss.backward() 
        optimizer.step() 

    return model

model = train_deep_model(df) # Entrenar el modelo una vez al inicio

# =========================
# Función para recomendar
# =========================

def get_recommendations(model, user_idx, N=5):
    model.eval() # Poner el modelo en modo de evaluación
    num_items = len(item_ids)# Poner el número de items (museos)
    user_tensor = torch.LongTensor([user_idx] * num_items) # Crear un tensor de usuarios con el índice del usuario seleccionado
    # Crear un tensor de items con todos los índices de items
    item_tensor = torch.LongTensor(list(range(num_items))) # Crear un tensor de items con todos los índices de items
    with torch.no_grad(): # Desactivar el cálculo de gradientes para la evaluación
        scores = model(user_tensor, item_tensor).numpy() # Obtener las puntuaciones del modelo para el usuario seleccionado

    top_indices = np.argsort(scores)[::-1][:N] # Obtener los índices de los N items con las puntuaciones más altas
    top_scores = scores[top_indices] # Obtener las puntuaciones correspondientes a esos índices

    return top_indices, top_scores 

# =========================
# Interfaz de usuario
# =========================

st.title("🎨 Recomendador de Museos en CDMX")

segmento_seleccionado = st.selectbox("Selecciona tu segmento:", list(user_ids.keys())) # Selección del segmento de usuario
user_idx = user_ids[segmento_seleccionado] # Obtener el índice del usuario seleccionado

if st.button("🔍 Ver recomendaciones"): 
    recomendaciones = get_recommendations(model, user_idx, N=5) # Obtener recomendaciones del modelo
    st.session_state['recomendaciones'] = recomendaciones # Guardar las recomendaciones en el estado de la sesión

if 'recomendaciones' in st.session_state: # Si ya hay recomendaciones en el estado de la sesión
    # Mostrar las recomendaciones
    item_ids_recom, scores = st.session_state['recomendaciones'] # Mostrar los IDs de los items recomendados y sus puntuaciones
    st.subheader("🎯 Museos recomendados:")

    opciones = [f"{reverse_item_ids[item_id]} — Score: {score:.2f}" for item_id, score in zip(item_ids_recom, scores)] # Crear una lista de opciones para el selectbox con los nombres de los museos y sus puntuaciones
    museo_seleccionado = st.selectbox("Selecciona un museo para ver más detalles:", opciones) # Selección del museo a mostrar

    seleccionado_index = opciones.index(museo_seleccionado) # Obtener el índice del museo seleccionado
    # Mostrar detalles del museo seleccionado
    item_id = item_ids_recom[seleccionado_index]
    nombre_museo = reverse_item_ids[item_id] # Obtener el nombre del museo seleccionado

    st.markdown(f"### 🏛️ {nombre_museo}") 
    st.markdown(f"**Puntaje estimado:** {scores[seleccionado_index]:.2f}") # Mostrar el puntaje estimado del museo seleccionado

    detalles = df[df['Centro de trabajo'] == nombre_museo].drop_duplicates() # Filtrar los detalles del museo seleccionado
    st.write(detalles[['Periodo', 'Estado', 'Tipo de sitio', 'Tipo de visitantes', 'Nacionalidad', 'Visitas']])

    # =====================
    # MAPA con folium
    # =====================
    lat = detalles['Latitud'].values[0] if 'Latitud' in detalles and not pd.isnull(detalles['Latitud'].values[0]) else None # Obtener la latitud del museo seleccionado
    lon = detalles['Longitud'].values[0] if 'Longitud' in detalles and not pd.isnull(detalles['Longitud'].values[0]) else None # Obtener la longitud del museo seleccionado
    # Mostrar la ubicación del museo en un mapa

    st.subheader("📍 Ubicación del museo")

    if lat is not None and lon is not None: # Si el museo tiene coordenadas geográficas
        st.write(f"**Ubicación:** {nombre_museo} ({lat}, {lon})") # Mostrar la ubicación del museo
        m = folium.Map(location=[lat, lon], zoom_start=15) # Crear un mapa centrado en las coordenadas del museo
        folium.Marker([lat, lon], popup=nombre_museo).add_to(m) # Añadir un marcador al mapa
        st_folium(m, width=700) # Mostrar el mapa interactivo en la aplicación Streamlit
    else:
        st.warning("Este museo no cuenta con coordenadas geográficas registradas.") 
        nombre_encoded = nombre_museo.replace(" ", "+") + "+CDMX" # Codificar el nombre del museo para la URL de Google Maps
        url = f"https://www.google.com/maps/search/{nombre_encoded}" # Crear la URL de búsqueda en Google Maps
        st.markdown(f"[🔎 Buscar ubicación en Google Maps]({url})", unsafe_allow_html=True)

    # =====================
    # Exportar a PDF
    # =====================
    st.subheader("📄 Exportar recomendación")

    if st.button("📥 Descargar como PDF"): 
        pdf = FPDF()  # Crear un objeto PDF
        pdf.add_page() # Añadir una página al PDF
        pdf.set_font("Arial", size=12) # Establecer la fuente del PDF
        # Añadir contenido al PDF
        pdf.cell(200, 10, txt=f"Recomendación de Museo", ln=1, align='C')
        pdf.ln(10) # Añadir un salto de línea
        pdf.multi_cell(0, 10, f"Museo recomendado: {nombre_museo}") # Añadir el nombre del museo recomendado
        pdf.multi_cell(0, 10, f"Puntaje: {scores[seleccionado_index]:.2f}") # Añadir el puntaje del museo recomendado
        pdf.multi_cell(0, 10, f"Segmento: {segmento_seleccionado}") # Añadir el segmento del usuario
        for col in ['Periodo', 'Estado', 'Tipo de sitio', 'Tipo de visitantes', 'Nacionalidad', 'Visitas']: # Añadir detalles del museo al PDF
            valor = detalles[col].values[0] # Obtener el valor de la columna correspondiente
            pdf.multi_cell(0, 10, f"{col}: {valor}") # Añadir el valor al PDF
        if lat is not None and lon is not None: # Si el museo tiene coordenadas geográficas
            pdf.multi_cell(0, 10, f"Ubicación: https://www.google.com/maps/search/?api=1&query={lat},{lon}") # Añadir la ubicación del museo al PDF
        else:
            pdf.multi_cell(0, 10, f"Ubicación: No disponible. Puedes buscar '{nombre_museo} CDMX' en Google Maps.") # Añadir un mensaje alternativo al PDF

        pdf_file = f"recomendacion_{nombre_museo.replace(' ', '_')}.pdf" # Nombre del archivo PDF
        pdf.output(pdf_file)

        with open(pdf_file, "rb") as f: # Abrir el archivo PDF en modo lectura binaria
            b64 = base64.b64encode(f.read()).decode() # Codificar el archivo PDF en base64
            href = f'<a href="data:application/pdf;base64,{b64}" download="{pdf_file}">📥 Haz clic aquí para descargar el PDF</a>'
            st.markdown(href, unsafe_allow_html=True) # Mostrar un enlace para descargar el PDF

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

