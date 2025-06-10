import pandas as pd
import time
import requests

# Función para corregir caracteres mal codificados
def corregir_caracteres(texto):
    if isinstance(texto, str):
        try:
            return texto.encode('latin1').decode('utf-8')
        except:
            return texto  # En caso de error, deja el texto original
    return texto

# Cargar el archivo CSV (ajustar encoding si es necesario)
df = pd.read_csv('museos_cdmx.csv', encoding='cp1252')

# Aplicar corrección de caracteres a la columna relevante
df['Centro de trabajo'] = df['Centro de trabajo'].apply(corregir_caracteres)

# Obtener nombres únicos de museos
museos_unicos = df['Centro de trabajo'].dropna().unique()

# Función para geolocalizar un museo usando OpenStreetMap/Nominatim
def geolocalizar(nombre):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": f"{nombre}, Ciudad de México, México",
        "format": "json",
        "limit": 1
    }
    headers = {
        "User-Agent": "MuseosCDMXApp/1.0 (contacto@ejemplo.com)"
    }
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200 and response.json():
            data = response.json()[0]
            return float(data["lat"]), float(data["lon"])
    except Exception as e:
        print(f"Error con {nombre}: {e}")
    return None, None

# Crear diccionario de coordenadas
coordenadas = {}
for museo in museos_unicos:
    print(f"Geolocalizando: {museo}")
    lat, lon = geolocalizar(museo)
    coordenadas[museo] = {"lat": lat, "lon": lon}
    time.sleep(1)  # Respetar el límite de la API

# Asignar coordenadas al DataFrame original
df['Latitud'] = df['Centro de trabajo'].map(lambda x: coordenadas.get(x, {}).get("lat"))
df['Longitud'] = df['Centro de trabajo'].map(lambda x: coordenadas.get(x, {}).get("lon"))

# Guardar el nuevo archivo CSV
df.to_csv("museos_cdmx_geolocalizado.csv", index=False, encoding="utf-8")
print("✅ Archivo guardado como museos_cdmx_geolocalizado.csv")
