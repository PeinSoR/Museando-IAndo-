import pandas as pd

# Cargar el CSV (asegúrate de que esté en la misma carpeta o da la ruta completa)
data = pd.read_csv('data.csv', encoding='latin1')

# Corregir caracteres mal codificados
data.columns = [col.strip() for col in data.columns]
data['Estado'] = data['Estado'].str.replace('MÃ©xico', 'México')
data['Nacionalidad'] = data['Nacionalidad'].str.replace('Ã', 'Á')

# Filtrar solo museos en Ciudad de México
filtro = (
    (data['Tipo de sitio'].str.strip() == 'Museo') &
    (data['Estado'].str.strip().isin(['Ciudad de México']))
)

# Aplicar filtro
museos_cdmx = data[filtro].copy()

# Guardar el resultado limpio en un nuevo CSV
museos_cdmx.to_csv('museos_cdmx_limpio.csv', index=False, encoding='utf-8')

print("Archivo 'museos_cdmx_limpio.csv' generado correctamente.")
