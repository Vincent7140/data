import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from skimage.transform import resize

# Charger les images
image_nerf = iio.imread("image_nerf.png").astype(np.float32) / 255.0
image_real = iio.imread("image_real.png").astype(np.float32) / 255.0

# Assure qu'on a des RGB uniquement
if image_nerf.shape[2] > 3:
    image_nerf = image_nerf[:, :, :3]
if image_real.shape[2] > 3:
    image_real = image_real[:, :, :3]

# Redimensionne image_real vers image_nerf si tailles différentes
if image_real.shape != image_nerf.shape:
    image_real = resize(image_real, image_nerf.shape, preserve_range=True, anti_aliasing=True)

# Calcul des différences RGB
diff = image_real - image_nerf
red = np.clip(diff, 0, 1)
green = np.clip(-diff, 0, 1)
blue = np.zeros_like(red)

# Image RGB rouge/vert
diff_rgb = np.stack([red, green, blue], axis=2)

# Affichage
plt.imshow(diff_rgb)
plt.title("Comparaison : Rouge = image réelle > NeRF, Vert = NeRF > image réelle")
plt.axis('off')
plt.show()
