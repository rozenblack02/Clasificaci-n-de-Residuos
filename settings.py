from pathlib import Path
import sys

# Obtener la ruta absoluta de este archivo
file_path = Path(__file__).resolve()
root_path = file_path.parent

# Agregar al sys.path si no está incluido
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Obtener la ruta relativa al directorio actual
ROOT = root_path.relative_to(Path.cwd())

# Fuentes de entrada
IMAGE = 'Imagen'
WEBCAM = 'Webcam'
SOURCES_LIST = [IMAGE, WEBCAM]

# Configuración de Imágenes
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'def.jfif'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'def1.jpg'

# Configuración de Videos
VIDEO_DIR = ROOT / 'videos'
VIDEO_1_PATH = VIDEO_DIR / 'video_1.mp4'
VIDEO_2_PATH = VIDEO_DIR / 'video_2.mp4'
VIDEO_3_PATH = VIDEO_DIR / 'video_3.mp4'
VIDEO_4_PATH = VIDEO_DIR / 'video_4.mp4'
VIDEO_5_PATH = VIDEO_DIR / 'video_5.mp4'
VIDEOS_DICT = {
    'video_1': VIDEO_1_PATH,
    'video_2': VIDEO_2_PATH,
    'video_3': VIDEO_3_PATH,
    'video_4': VIDEO_4_PATH,
    'video_5': VIDEO_5_PATH,
}

# Configuración del modelo de ML
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'best.pt'
#SEGMENTATION_MODEL = MODEL_DIR / 'modelo_segmentacion.pt'

# Webcam
WEBCAM_PATH = 0
