# app.py traducido y modificado para incluir carga de videos
from pathlib import Path
import PIL
import streamlit as st
import settings
import helper

# Configuración de la página
st.set_page_config(
    page_title="Clasificación de Residuos con YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("Clasificación de Residuos con YOLOv8")

# Barra lateral - configuración del modelo
st.sidebar.header("Configuración del Modelo ML")
model_type = st.sidebar.radio("Selecciona la tarea", ['Detección'])

confidence = float(st.sidebar.slider(
    "Selecciona la confianza del modelo", 25, 100, 40)) / 100

if model_type == 'Detección':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentación':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Cargar modelo
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"No se pudo cargar el modelo desde la ruta: {model_path}")
    st.error(ex)

# Barra lateral - configuración de entrada
st.sidebar.header("Configuración de Entrada")

# Fuentes disponibles
source_radio = st.sidebar.radio("Selecciona la fuente", ['Imagen', 'Video', 'Webcam'])

source_img = None

# Si se selecciona una imagen
if source_radio == 'Imagen':
    source_img = st.sidebar.file_uploader("Selecciona una imagen...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Imagen por defecto", use_container_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Imagen cargada", use_container_width=True)
        except Exception as ex:
            st.error("Error al abrir la imagen")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(default_detected_image_path)
            st.image(default_detected_image_path, caption='Resultado por defecto', use_container_width=True)
        else:
            if st.sidebar.button('Detectar objetos'):
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Resultado de detección', use_container_width=True)
                try:
                    with st.expander("Resultados detallados"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    st.write("No hay imagen cargada")

elif source_radio == 'Webcam':
    helper.play_webcam(confidence, model)

elif source_radio == 'Video':
    helper.play_stored_video(confidence, model)

else:
    st.error("Selecciona un tipo de fuente válido")
