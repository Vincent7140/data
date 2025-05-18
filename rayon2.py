import numpy as np
import torch

def generate_virtual_rays(cam_pos, look_at, H, W, fov_deg=40, near=0.0, far=1.0):
    """
    Génére des rayons depuis une caméra virtuelle pinhole regardant vers 'look_at'.
    Args:
        cam_pos: (3,) np.array, position de la caméra en coordonnées ECEF ou normalisées
        look_at: (3,) np.array, point visé par la caméra
        H, W: taille de l'image synthétique
        fov_deg: champ de vision vertical en degrés
        near, far: bornes de profondeur
    Return:
        torch.Tensor (H*W, 8): rayons [origin (3), direction (3), near, far]
    """
    cam_pos = np.array(cam_pos)
    look_at = np.array(look_at)

    # Orientation
    forward = look_at - cam_pos
    forward /= np.linalg.norm(forward)
    up = np.array([0, 0, 1], dtype=np.float32)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    # Grille d'image
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    i = i.reshape(-1)
    j = j.reshape(-1)

    aspect_ratio = W / H
    fov_rad = np.radians(fov_deg)
    x = (2 * (i + 0.5) / W - 1) * np.tan(fov_rad / 2) * aspect_ratio
    y = (1 - 2 * (j + 0.5) / H) * np.tan(fov_rad / 2)

    # Direction de chaque rayon
    directions = x[:, None] * right + y[:, None] * up + forward[None, :]
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    origins = np.tile(cam_pos[None, :], (H * W, 1))
    nears = np.full((H * W, 1), near, dtype=np.float32)
    fars = np.full((H * W, 1), far, dtype=np.float32)

    rays = np.hstack([origins, directions, nears, fars])
    return torch.from_numpy(rays).float()

from utils.virtual_camera import generate_virtual_rays

# Paramètres de la caméra virtuelle
H, W = 128, 128
fov = 40
center = dataset.center.cpu().numpy()
range_ = dataset.range.cpu().numpy()

# Exemple : une orbite circulaire de rayon 0.3 autour de la scène
for i in range(4):
    angle = 2 * np.pi * i / 4
    x = center[0] + 0.3 * np.cos(angle)
    y = center[1] + 0.3 * np.sin(angle)
    z = center[2] + 0.1  # un peu au-dessus de la scène

    cam_pos = np.array([x, y, z])
    rays = generate_virtual_rays(cam_pos, center, H, W, fov)
    rays = rays.cuda()

    with torch.no_grad():
        result = batched_inference(models, rays, ts=None, args=args)
        rgb = result["rgb_fine" if "rgb_fine" in result else "rgb_coarse"]
        rgb = rgb.view(H, W, 3).permute(2, 0, 1).cpu()

    # Sauvegarde
    save_path = os.path.join(out_dir, f"virtual_view_{i:02d}.tif")
    train_utils.save_output_image(rgb, save_path, src_path=None)
