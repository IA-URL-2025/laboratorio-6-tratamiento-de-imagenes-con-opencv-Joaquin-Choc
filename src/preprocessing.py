import cv2
import numpy as np


def to_grayscale(image):
    """
    Convierte la imagen a escala de grises.
    """
    # Utilizar cv2.cvtColor con el código de conversión BGR2GRAY
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_image(image, width, height):
    """
    Redimensiona la imagen a las dimensiones indicadas.
    """
    # Utilizar cv2.resize con interpolación cv2.INTER_AREA
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def apply_blur(image, kernel_size=5):
    """
    Aplica un filtro de suavizado Gaussiano a la imagen.
    """
    # Utilizar cv2.GaussianBlur con el kernel_size proporcionado
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    """
    Ajusta el brillo y el contraste de la imagen.
    """
    # Utilizar cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def apply_threshold(image, thresh_value=127):
    """
    Aplica umbralización binaria a una imagen en escala de grises.
    """
    if len(image.shape) != 2:
        raise ValueError("apply_threshold requiere una imagen en escala de grises (1 canal).")
    
    # Utilizar cv2.threshold con tipo cv2.THRESH_BINARY y el valor máximo 255
    _, binarized_image = cv2.threshold(image, thresh_value, 255, cv2.THRESH_BINARY)
    return binarized_image


def detect_edges(image, low=50, high=150):
    """
    Detecta bordes con el algoritmo de Canny.
    """
    # Primero convertir la imagen a escala de grises si tiene más de un canal
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Aplicar cv2.Canny con los umbrales low y high
    return cv2.Canny(gray, low, high)


def full_pipeline(image, target_width=224, target_height=224):
    """
    Ejecuta el pipeline completo de preprocesamiento.
    """
    # 1) Redimensionar 
    img_resized = resize_image(image, target_width, target_height)
    
    # 2) Convertir a escala de grises 
    img_gray = to_grayscale(img_resized)
    
    # 3) Aplicar blur (kernel = 3) 
    img_blurred = apply_blur(img_gray, kernel_size=3)
    
    # 4) Detectar bordes (low=50, high=150) 
    img_edges = detect_edges(img_blurred, low=50, high=150)
    
    # Retornar la imagen final procesada.
    return img_edges