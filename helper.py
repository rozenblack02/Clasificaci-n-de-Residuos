from ultralytics import YOLO
import streamlit as st
import cv2
import settings


def load_model(model_path):
    """
    Carga un modelo YOLO desde la ruta especificada.
    """
    model = YOLO(model_path)
    return model


def mostrar_opciones_tracker():
    mostrar_tracker = st.radio("Mostrar seguimiento", ('Sí', 'No'))
    usar_tracker = True if mostrar_tracker == 'Sí' else False
    if usar_tracker:
        tipo_tracker = st.radio("Tipo de tracker", ("bytetrack.yaml", "botsort.yaml"))
        return usar_tracker, tipo_tracker
    return usar_tracker, None


def mostrar_detecciones(conf, modelo, st_frame, imagen, usar_tracker=None, tipo_tracker=None):
    """
    Muestra los objetos detectados sobre una imagen o cuadro de video.
    """
    imagen = cv2.resize(imagen, (720, int(720*(9/16))))

    if usar_tracker:
        res = modelo.track(imagen, conf=conf, persist=True, tracker=tipo_tracker)
    else:
        res = modelo.predict(imagen, conf=conf)

    resultado = res[0].plot()
    st_frame.image(resultado,
                  caption='Video Detectado',
                  channels="BGR",
                  use_column_width=True)


def play_webcam(conf, modelo):
    """
    Usa la webcam para detectar objetos en tiempo real con YOLOv8.
    """
    camara = settings.WEBCAM_PATH
    usar_tracker, tipo_tracker = mostrar_opciones_tracker()
    if st.sidebar.button('Detectar residuos en webcam'):
        try:
            cap = cv2.VideoCapture(camara)
            st_frame = st.empty()
            while cap.isOpened():
                success, frame = cap.read()
                if success:
                    mostrar_detecciones(conf, modelo, st_frame, frame, usar_tracker, tipo_tracker)
                else:
                    cap.release()
                    break
        except Exception as e:
            st.sidebar.error(f"Error cargando video: {e}")


def play_stored_video(conf, modelo):
    """
    Permite al usuario cargar un video y aplicar detección de objetos en él.
    """
    video_file = st.sidebar.file_uploader("Sube un video...", type=["mp4", "mov", "avi", "mkv"])

    usar_tracker, tipo_tracker = mostrar_opciones_tracker()

    if video_file is not None:
        st.video(video_file)

        if st.sidebar.button('Detectar residuos en video'):
            try:
                with open("temp_video.mp4", "wb") as out:
                    out.write(video_file.read())

                cap = cv2.VideoCapture("temp_video.mp4")
                st_frame = st.empty()

                while cap.isOpened():
                    success, frame = cap.read()
                    if success:
                        mostrar_detecciones(conf, modelo, st_frame, frame, usar_tracker, tipo_tracker)
                    else:
                        cap.release()
                        break

            except Exception as e:
                st.sidebar.error(f"Error al procesar el video: {e}")
