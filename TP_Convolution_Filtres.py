import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random
from PIL import Image

def apply_convolution(image, kernel):
    """
    Applique une convolution à une image (en niveaux de gris ou RGB) à l’aide d’un noyau (kernel).
    Retourne l’image filtrée avec les valeurs limitées à [0, 255] et convertie en uint8.
    """

    # Si l'image a 3 dimensions → image en couleur (R, G, B)
    if len(image.shape) == 3:
        output = np.zeros_like(image)  # Initialiser une image de sortie vide
        for c in range(image.shape[2]):  # Appliquer la convolution sur chaque canal (0: R, 1: G, 2: B)
            output[:, :, c] = convolve_channel(image[:, :, c], kernel)
    else:
        # Image en niveaux de gris
        output = convolve_channel(image, kernel)

    # Clipper les valeurs dans [0, 255] et convertir en image 8 bits
    return np.clip(output, 0, 255).astype(np.uint8)


def convolve_channel(image, kernel):
    """
    Applique la convolution sur une image à un seul canal (2D) avec un filtre 3x3.
    Utilise un padding 'constant' avec 0 par défaut (marges noires).
    """

    # Ajouter une bordure de 1 pixel (padding) autour de l'image
    padded = np.pad(image, ((1, 1), (1, 1)), mode='constant')

    # Créer une image de sortie de même taille que l'image d'origine
    output = np.zeros_like(image, dtype=np.float32)

    # Parcourir tous les pixels de l’image originale (hors bordure)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extraire la région 3x3 centrée sur le pixel (i, j)
            region = padded[i:i+3, j:j+3]
            # Produit élément par élément entre la région et le noyau, puis somme
            output[i, j] = np.sum(region * kernel)
    return output

def load_image(image_path, channel=1):
    """
    Charge une image en niveaux de gris ou RGB avec vérification automatique du type
    Args:
        image_path: chemin vers l'image
        channel: 1 pour niveaux de gris, 3 pour RGB (par défaut: 1)
    Returns:
        L'image chargée
    """
    if channel == 1:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        assert image is not None, "Erreur: Impossible de charger l'image en niveaux de gris"
        assert len(image.shape) == 2, "L'image n'est pas en niveaux de gris comme attendu"
    elif channel == 3:
        image = cv2.imread(image_path)
        assert image is not None, "Erreur: Impossible de charger l'image RGB"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        assert len(image.shape) == 3 and image.shape[2] == 3, "L'image n'est pas RGB comme attendu"
    else:
        raise ValueError("Le paramètre channel doit être 1 (niveaux de gris) ou 3 (RGB)")
    
    return image

def convolve_channel(image, kernel):
    """
    Applique la convolution sur un seul canal d'image
    Args:
        image: image 2D (un seul canal)
        kernel: noyau de convolution 2D
    Returns:
        Image filtrée
    """
    # Vérifications des dimensions
    assert isinstance(image, np.ndarray), "L'image doit être un tableau NumPy"
    assert isinstance(kernel, np.ndarray), "Le noyau doit être un tableau NumPy"
    assert len(kernel.shape) == 2, "Le noyau doit être une matrice 2D"
    assert kernel.shape[0] == kernel.shape[1], "Le noyau doit être carré"
    assert kernel.shape[0] % 2 == 1, "Le noyau doit avoir une taille impaire"
    assert len(image.shape) == 2, "L'image doit être 2D pour un seul canal"
    assert image.shape[0] >= kernel.shape[0], "L'image est trop petite pour le noyau en hauteur"
    assert image.shape[1] >= kernel.shape[1], "L'image est trop petite pour le noyau en largeur"

    # Dimensions
    img_h, img_w = image.shape
    kernel_h, kernel_w = kernel.shape
    pad_h, pad_w = kernel_h // 2, kernel_w // 2

    # Ajout du padding
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(image, dtype=np.float32)

    # Application de la convolution
    for i in range(img_h):
        for j in range(img_w):
            region = padded[i:i+kernel_h, j:j+kernel_w]
            output[i, j] = np.sum(region * kernel)
    
    return output

def apply_convolution(image, kernel):
    """
    Applique la convolution sur une image (niveaux de gris ou RGB)
    Args:
        image: image d'entrée (2D ou 3D)
        kernel: noyau de convolution
    Returns:
        Image filtrée
    """
    # Vérifications des entrées
    assert isinstance(image, np.ndarray), "L'image doit être un tableau NumPy"
    assert isinstance(kernel, np.ndarray), "Le noyau doit être un tableau NumPy"
    assert len(kernel.shape) == 2, "Le noyau doit être une matrice 2D"
    assert kernel.shape[0] == kernel.shape[1], "Le noyau doit être carré"
    assert kernel.shape[0] % 2 == 1, "Le noyau doit avoir une taille impaire"
    assert len(image.shape) in [2, 3], "L'image doit être en niveaux de gris (2D) ou RGB (3D)"
    
    if len(image.shape) == 3:  # Image RGB
        assert image.shape[2] == 3, "L'image RGB doit avoir 3 canaux"
        output = np.zeros_like(image, dtype=np.float32)
        for c in range(3):
            output[:,:,c] = convolve_channel(image[:,:,c], kernel)
    else:  # Image niveaux de gris
        output = convolve_channel(image, kernel)
    
    # Normalisation et conversion en uint8
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output

def display_results(original, filtered_images, titles):
    """
    Affiche l'image originale et les images filtrées
    Args:
        original: image originale
        filtered_images: liste des images filtrées
        titles: liste des titres pour chaque image filtrée
    """
    n = len(filtered_images) + 1
    plt.figure(figsize=(15, 5))
    
    # Afficher l'image originale
    plt.subplot(1, n, 1)
    plt.title("Image originale")
    if len(original.shape) == 2:
        plt.imshow(original, cmap='gray')
    else:
        plt.imshow(original)
    plt.axis('off')
    
    # Afficher les images filtrées
    for i in range(2, n+1):
        plt.subplot(1, n, i)
        plt.title(titles[i-2])
        if len(filtered_images[i-2].shape) == 2:
            plt.imshow(filtered_images[i-2], cmap='gray')
        else:
            plt.imshow(filtered_images[i-2])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def generate_random_kernel(size, seed=None):
    """
    Génère un noyau aléatoire normalisé
    Args:
        size: taille du noyau (doit être impair)
        seed: seed pour la reproductibilité
    Returns:
        Noyau aléatoire normalisé
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    assert size % 2 == 1, "La taille du noyau doit être impaire"
    kernel = np.random.rand(size, size)
    kernel = kernel / np.sum(kernel)  # Normalisation
    return kernel

# Définition des noyaux de convolution
# 1. Filtre de flou (moyenne)
blur_kernel_3x3 = np.array([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
])

blur_kernel_5x5 = np.ones((5, 5)) / 25
blur_kernel_7x7 = np.ones((7, 7)) / 49
# 2. Filtre Sobel horizontal (détection de contours horizontaux)
sobel_horizontal = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# 3. Filtre Sobel vertical (détection de contours verticaux)
sobel_vertical = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])

# 4. Filtre de netteté (sharpening)
sharp_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

# 5. Filtre de relief (emboss)
emboss_kernel = np.array([
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]
])

# 6. Filtres aléatoires
random_kernel_3x3 = generate_random_kernel(3, seed=42)
random_kernel_5x5 = generate_random_kernel(5, seed=42)
random_kernel_7x7 = generate_random_kernel(7, seed=42)

# Liste des filtres et leurs noms
kernels = [
    (blur_kernel_3x3, "Flou 3x3"),
    (blur_kernel_5x5, "Flou 5x5"),
    (blur_kernel_7x7, "Flou 7x7"),
    (sobel_horizontal, "Sobel Horizontal"),
    (sobel_vertical, "Sobel Vertical"),
    (sharp_kernel, "Netteté"),
    (emboss_kernel, "Relief"),
    (random_kernel_3x3, "Aléatoire 3x3"),
    (random_kernel_5x5, "Aléatoire 5x5"),
    (random_kernel_7x7, "Aléatoire 7x7")
]


# Charger les images
image_path = 'boats.pgm'
image= Image.open(image_path)

# Convert to numpy array and check shape
image_array = np.array(image)
image_mode = image.mode
image_shape = image_array.shape

image_mode, image_shape



gray_image = load_image(image_path, channel=1)
rgb_image = load_image(image_path, channel=3)
# Sauvegarder les résultats
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)


if  image_mode == "L" :
    # Appliquer les filtres sur l'image en niveaux de gris
    gray_filtered = []
    gray_titles = []
    for kernel, name in kernels:
        gray_filtered.append(apply_convolution(gray_image, kernel))
        gray_titles.append(name)
    # Afficher les résultats pour les images en niveaux de gris
    print("Résultats pour l'image en niveaux de gris:")
    display_results(gray_image, gray_filtered, gray_titles)
    # Sauvegarder les résultats
    for i, (filtered, title) in enumerate(zip(gray_filtered, gray_titles)):
        filename = os.path.join(output_dir, f"gray_{title.replace(' ', '_').lower()}.jpg")
        cv2.imwrite(filename, filtered)
else:
    # Appliquer les filtres sur l'image RGB
    rgb_filtered = []
    rgb_titles = []
    for kernel, name in kernels:
        rgb_filtered.append(apply_convolution(rgb_image, kernel))
        rgb_titles.append(name)
    # Afficher les résultats pour les images RGB
    print("Résultats pour l'image RGB:")
    display_results(rgb_image, rgb_filtered, rgb_titles)
    # Sauvegarder les résultats
    for i, (filtered, title) in enumerate(zip(rgb_filtered, rgb_titles)):
        filename = os.path.join(output_dir, f"rgb_{title.replace(' ', '_').lower()}.jpg")
        cv2.imwrite(filename, cv2.cvtColor(filtered, cv2.COLOR_RGB2BGR))
print("Les images filtrées ont été sauvegardées dans le dossier 'output'")