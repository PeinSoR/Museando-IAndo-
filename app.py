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
import requests
from datetime import datetime
import time
from PIL import Image
import io
import sys


# Configuración de la página
st.set_page_config(page_title="Recomendador de Museos CDMX", page_icon="🏛️", layout="wide")

# =========================
# Cargar y procesar datos
# =========================

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("museos_cdmx_geolocalizado.csv", encoding="utf-8")
        df.columns = df.columns.str.strip()
        df['Segmento'] = df['Nacionalidad'] + " - " + df['Tipo de visitantes']
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# Mapeos de IDs
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

    top_indices = np.argsort(scores)[::-1][:N]
    top_scores = scores[top_indices]

    return top_indices, top_scores

# =========================
# Función mejorada para obtener información de museos
# =========================

@st.cache_data(ttl=86400)  # Cache por 24 horas
def get_museum_info(museo_nombre, lat=None, lon=None, api_key=None):
    """
    Obtiene información completa del museo incluyendo horarios en español y fotos
    """
    if not api_key:
        return {
            "horario": "🔑 Configura tu API Key de Google Places",
            "fotos": []
        }
    
    try:
        # Diccionario de horarios manuales para museos conocidos (en español)
        HORARIOS_MANUALES = {
            "Museo Nacional de Antropología": {
                "horario": "Martes a domingo de 9:00 a 19:00",
                "fotos": [
                    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/MNA_View.jpg/800px-MNA_View.jpg",
                    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/Museo_Nacional_de_Antropolog%C3%ADa_%28M%C3%A9xico%29.jpg/800px-Museo_Nacional_de_Antropolog%C3%ADa_%28M%C3%A9xico%29.jpg"
                ]
            },
            "Palacio de Bellas Artes": {
                "horario": "Martes a domingo de 10:00 a 18:00",
                "fotos": [
                    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/Mexico_City_-_Palacio_de_Bellas_Artes_2.jpg/800px-Mexico_City_-_Palacio_de_Bellas_Artes_2.jpg",
                    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Palacio_de_Bellas_Artes_2013.jpg/800px-Palacio_de_Bellas_Artes_2013.jpg"
                ]
            },
            "Museo del Templo Mayor": {
                "horario": "Martes a domingo de 9:00 a 17:00",
                "fotos": [
                    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5d/TemploMayor1.jpg/800px-TemploMayor1.jpg",
                    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Museo_del_Templo_Mayor_2013.jpg/800px-Museo_del_Templo_Mayor_2013.jpg"
                ]
            },
            "Museo Soumaya": {
                "horario": "Lunes a domingo de 10:30 a 18:30",
                "fotos": [
                    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/Museo_Soumaya_%28noche%29.jpg/800px-Museo_Soumaya_%28noche%29.jpg",
                    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Museo_Soumaya_Plaza_Carso_2013.jpg/800px-Museo_Soumaya_Plaza_Carso_2013.jpg"
                ]
            }
        }
        
        # Primero verifica si está en nuestros datos manuales
        if museo_nombre in HORARIOS_MANUALES:
            return HORARIOS_MANUALES[museo_nombre]
        
        # Paso 1: Búsqueda exacta por nombre
        url_text_search = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={museo_nombre} CDMX&language=es&key={api_key}"
        response = requests.get(url_text_search, timeout=10)
        results = response.json()
        
        if results.get('results') and len(results['results']) > 0:
            # Buscamos la mejor coincidencia de nombre
            best_match = None
            for place in results['results']:
                if museo_nombre.lower() in place.get('name', '').lower():
                    best_match = place
                    break
            
            place = best_match or results['results'][0]
            place_id = place['place_id']
            
            # Obtenemos detalles completos (incluyendo fotos)
            details_url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields=name,opening_hours,formatted_address,url,geometry,photos&key={api_key}&language=es"
            details_response = requests.get(details_url)
            details = details_response.json()
            
            if 'result' in details:
                result = details['result']
                info = {
                    "nombre": result.get('name', museo_nombre),
                    "horario": "Horario no disponible",
                    "direccion": result.get('formatted_address', 'Dirección no disponible'),
                    "url": result.get('url', ''),
                    "fotos": []
                }
                
                # Procesar horarios en español
                if 'opening_hours' in result:
                    opening_hours = result['opening_hours']
                    if opening_hours.get('weekday_text'):
                        info['horario'] = "\n".join(opening_hours['weekday_text'])
                
                # Obtener fotos del lugar (máx 5)
                if 'photos' in result:
                    for photo in result['photos'][:5]:  # Limitar a 5 fotos
                        photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=800&photoreference={photo['photo_reference']}&key={api_key}"
                        info['fotos'].append(photo_url)
                
                return info
        
        # Paso 2: Si la búsqueda por nombre falló, intentamos con coordenadas
        if lat and lon:
            url_nearby = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lon}&radius=100&keyword={museo_nombre}&language=es&key={api_key}"
            response = requests.get(url_nearby, timeout=10)
            results = response.json()
            
            if results.get('results') and len(results['results']) > 0:
                place = min(
                    results['results'],
                    key=lambda x: (
                        abs(x['geometry']['location']['lat'] - lat) + 
                        abs(x['geometry']['location']['lng'] - lon)
                ))
                
                place_id = place['place_id']
                details_url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields=name,opening_hours,formatted_address,url,photos&key={api_key}&language=es"
                details_response = requests.get(details_url)
                details = details_response.json()
                
                if 'result' in details:
                    result = details['result']
                    info = {
                        "nombre": result.get('name', museo_nombre),
                        "horario": "Horario cercano no disponible",
                        "direccion": result.get('formatted_address', 'Dirección cercana no disponible'),
                        "url": result.get('url', ''),
                        "fotos": []
                    }
                    
                    if 'opening_hours' in result:
                        opening_hours = result['opening_hours']
                        if opening_hours.get('weekday_text'):
                            info['horario'] = "Horario cercano:\n" + "\n".join(opening_hours['weekday_text'])
                    
                    if 'photos' in result:
                        for photo in result['photos'][:5]:
                            photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=800&photoreference={photo['photo_reference']}&key={api_key}"
                            info['fotos'].append(photo_url)
                    
                    return info
        
        return {
            "horario": "Horario no disponible. Consulta directamente en Google Maps.",
            "fotos": []
        }
    
    except Exception as e:
        print(f"Error obteniendo información: {e}")
        return {
            "horario": "Error al obtener información. Intenta más tarde.",
            "fotos": []
        }

# =========================
# Interfaz de usuario mejorada
# =========================

st.title("🎨 Recomendador de Museos en CDMX")

# Configuración de API Key desde secrets
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except Exception as e:
    st.error("🔑 Error al cargar la API Key. Verifica tu archivo secrets.toml")
    st.stop()

with st.sidebar.expander("ℹ️ Información adicional"):
    st.write("""
    Este sistema recomienda museos basado en el historial de visitas por segmento.
    Los segmentos combinan nacionalidad y tipo de visitantes.
    """)
    st.write("📧 Contacto: museandoiando@gmail.com")

segmento_seleccionado = st.selectbox("Selecciona tu segmento:", list(user_ids.keys()))
user_idx = user_ids[segmento_seleccionado]

if st.button("🔍 Ver recomendaciones"):
    with st.spinner('Buscando recomendaciones...'):
        recomendaciones = get_recommendations(model, user_idx, N=5)
        st.session_state['recomendaciones'] = recomendaciones

if 'recomendaciones' in st.session_state:
    item_ids_recom, scores = st.session_state['recomendaciones']
    
    st.subheader("🎯 Museos recomendados:")
    opciones = [f"{reverse_item_ids[item_id]} — Score: {score:.2f}" 
               for item_id, score in zip(item_ids_recom, scores)]
    
    museo_seleccionado = st.selectbox("Selecciona un museo para ver más detalles:", opciones)
    seleccionado_index = opciones.index(museo_seleccionado)
    item_id = item_ids_recom[seleccionado_index]
    nombre_museo = reverse_item_ids[item_id]

    # Mostrar detalles del museo en columnas
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"### 🏛️ {nombre_museo}")
        
        # Mostrar puntaje con estrellas
        score = scores[seleccionado_index]
        scaled_score = max(0, min(score, 5))
        st.markdown(f"**Puntaje estimado:** {'🏛️' * int(round(scaled_score))} ({scaled_score:.2f}/5)")
        
        detalles = df[df['Centro de trabajo'] == nombre_museo].drop_duplicates()
        st.write(detalles[['Periodo', 'Estado', 'Tipo de sitio', 'Tipo de visitantes', 'Nacionalidad', 'Visitas']])

    # En la sección donde muestras la información del museo (reemplaza todo el bloque de horarios):
    with col2:
        # Obtener coordenadas
        lat = detalles['Latitud'].values[0] if 'Latitud' in detalles and not pd.isnull(detalles['Latitud'].values[0]) else None
        lon = detalles['Longitud'].values[0] if 'Longitud' in detalles and not pd.isnull(detalles['Longitud'].values[0]) else None
        
        with st.spinner('Obteniendo información actualizada...'):
            museo_info = get_museum_info(nombre_museo, lat, lon, api_key)
            
            # Mostrar horario con estilo mejorado
            st.markdown("""
            <style>
                .horario-box {
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 10px;
                    margin: 10px 0;
                    border-left: 4px solid #4e73df;
                }
                .horario-item {
                    margin: 5px 0;
                }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown("⏰ **Horario**")
            with st.container():
                horario_html = ''.join(
                    f'<div style="color: #393D42;" class="horario-item">{line}</div>'
                    for line in museo_info['horario'].split('\n')
                )
                st.markdown(f"""
                <div class="horario-box">
                    {horario_html}
                </div>
            """, unsafe_allow_html=True)

            
            st.markdown(f"📍 **Dirección:**\n{museo_info.get('direccion', 'No disponible')}")
            if museo_info.get('url'):
                st.markdown(f"🗺️ [Ver en Google Maps]({museo_info['url']})")
    # Carrusel de imágenes
    if museo_info.get('fotos'):
        st.subheader("📷 Galeria de imágenes del museo")
        cols = st.columns(3)
        for i, foto_url in enumerate(museo_info['fotos']):
            cols[i % 3].image(foto_url, use_container_width=True)

    # Mapa de ubicación
    st.subheader("📍 Ubicación exacta")
    if lat and lon:
        m = folium.Map(location=[lat, lon], zoom_start=16)
        folium.Marker(
            [lat, lon],
            popup=nombre_museo,
            tooltip="Ver detalles",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)
        st_folium(m, width=800, height=500)
    else:
        st.warning("Coordenadas no disponibles")
        if museo_info.get('url'):
            st.markdown(f"🔍 [Buscar en Google Maps]({museo_info['url']})")
        else:
            nombre_encoded = nombre_museo.replace(" ", "+") + "+CDMX"
            st.markdown(f"🔍 [Buscar en Google Maps](https://www.google.com/maps/search/{nombre_encoded})")


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