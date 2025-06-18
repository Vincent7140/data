import numpy as np
import matplotlib.pyplot as plt
import rasterio
from skimage.transform import resize

# Lire image NeRF (plus petite)
with rasterio.open("image_nerf.tif") as src_nerf:
    image_nerf = src_nerf.read([1, 2, 3]).transpose(1, 2, 0).astype(np.float32)

# Lire image réelle
with rasterio.open("image_real.tif") as src_real:
    image_real = src_real.read([1, 2, 3]).transpose(1, 2, 0).astype(np.float32)

# Redimensionner image réelle vers image_nerf
if image_real.shape != image_nerf.shape:
    image_real = resize(image_real, image_nerf.shape, preserve_range=True, anti_aliasing=True)

# Normalisation
image_real /= image_real.max()
image_nerf /= image_nerf.max()

# Calcul différence
diff = image_real - image_nerf
red = np.clip(diff, 0, 1)
green = np.clip(-diff, 0, 1)
blue = np.zeros_like(red)

# Empilement RGB
diff_rgb = np.stack([red, green, blue], axis=2)

# Affichage
plt.imshow(diff_rgb)
plt.title("Comparaison : Rouge = image_real > image_nerf, Vert = image_nerf > image_real")
plt.axis('off')
plt.show()
