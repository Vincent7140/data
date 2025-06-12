def video_rotation_elevation(model, arg_dict, hwf, radius, az_deg_fixed=180, el_range=(30, 120), n_views=60, out_path='elevation_video.mp4'):
    """
    Génère une vidéo en faisant varier l'élévation (vue du dessus vers le dessous).

    model     : modèle S-NeRF
    arg_dict  : dictionnaire de config
    hwf       : (H, W, focal)
    radius    : rayon (distance caméra-scène)
    az_deg_fixed : azimuth constant en degrés
    el_range  : tuple (min_deg, max_deg) pour l’élévation
    n_views   : nombre d'images (frames)
    out_path  : chemin de la vidéo en sortie
    """
    az = np.deg2rad(az_deg_fixed)
    elevations = np.linspace(*el_range, n_views)
    
    frames = []
    for el_deg in elevations:
        el = np.deg2rad(el_deg)
        pose = pose_spherical(az, el, radius)

        view_dir = tf.reshape(tf.convert_to_tensor([az, el], dtype=tf.float32), [1,2])
        light_dir = view_dir

        result = render_image(model, arg_dict, hwf, pose, 1.0, light_dir, view_dir, rets=['rgb'])
        rgb_image = (255 * result["rgb"].numpy()).astype(np.uint8)
        frames.append(rgb_image)

    imageio.mimwrite(out_path, frames, fps=10, quality=8)
    print(f"[✔] Vidéo elevation sauvegardée : {out_path}")


def video_rotation_elevation(model, arg_dict, hwf, radius, az_deg_fixed=180, el_range=(30, 120), n_views=60, out_path='elevation_video.mp4'):
    """
    Génère une vidéo en faisant varier l'élévation (vue du dessus vers le dessous).

    model     : modèle S-NeRF
    arg_dict  : dictionnaire de config
    hwf       : (H, W, focal)
    radius    : rayon (distance caméra-scène)
    az_deg_fixed : azimuth constant en degrés
    el_range  : tuple (min_deg, max_deg) pour l’élévation
    n_views   : nombre d'images (frames)
    out_path  : chemin de la vidéo en sortie
    """
    az = np.deg2rad(az_deg_fixed)
    elevations = np.linspace(*el_range, n_views)
    
    frames = []
    for el_deg in elevations:
        el = np.deg2rad(el_deg)
        pose = pose_spherical(az, el, radius)

        view_dir = tf.reshape(tf.convert_to_tensor([az, el], dtype=tf.float32), [1,2])
        light_dir = view_dir

        result = render_image(model, arg_dict, hwf, pose, 1.0, light_dir, view_dir, rets=['rgb'])
        rgb_image = (255 * result["rgb"].numpy()).astype(np.uint8)
        frames.append(rgb_image)

    imageio.mimwrite(out_path, frames, fps=10, quality=8)
    print(f"[✔] Vidéo elevation sauvegardée : {out_path}")


video_rotation_azimuth(model, arg_dict, hwf, radius=617000.0/0.3)
video_rotation_elevation(model, arg_dict, hwf, radius=617000.0/0.3)


import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

def annotate_image(image_np, az_deg, el_deg, radius):
    """
    Annoter une image numpy avec texte affichant azimuth, elevation, radius.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image_np)
    ax.axis('off')
    ax.set_title(f'Az: {az_deg:.1f}°, El: {el_deg:.1f}°, R: {radius:.1f}', fontsize=10)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    annotated_image = np.array(Image.open(buf))
    return annotated_image



import numpy as np
import tensorflow as tf
import imageio
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

def annotate_image(image_np, az_deg, el_deg, radius):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image_np)
    ax.axis('off')
    ax.set_title(f'Az: {az_deg:.1f}°, El: {el_deg:.1f}°, R: {radius:.1f}', fontsize=10)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf))

def video_converge_to_z_axis(model, arg_dict, hwf, az_deg=180, el_deg_fixed=60, radius_start=617000.0/0.3, radius_end=None, n_frames=60, out_path='converge_z.mp4'):
    """
    Génère une vidéo où la caméra se rapproche de l’axe Z, sans changer d’altitude (z constant).

    model        : modèle S-NeRF
    arg_dict     : dictionnaire de configuration
    hwf          : (H, W, focal)
    az_deg       : azimuth constant
    el_deg_fixed : élévation de départ en degrés
    radius_start : rayon de départ
    radius_end   : rayon de fin (défaut = moitié du départ)
    n_frames     : nombre d’images
    out_path     : chemin de sortie de la vidéo
    """
    if radius_end is None:
        radius_end = radius_start * 0.5  # par défaut : on va 2x plus près

    z_fixed = radius_start * np.sin(np.deg2rad(el_deg_fixed))
    az = np.deg2rad(az_deg)
    
    radii = np.linspace(radius_start, radius_end, n_frames)
    frames = []

    for r in radii:
        el = np.arcsin(z_fixed / r)
        pose = pose_spherical(az, el, r)
        view_dir = tf.reshape(tf.convert_to_tensor([az, el], dtype=tf.float32), [1,2])
        light_dir = view_dir

        result = render_image(model, arg_dict, hwf, pose, 1.0, light_dir, view_dir, rets=['rgb'])
        rgb_image = (255 * result["rgb"].numpy()).astype(np.uint8)
        annotated = annotate_image(rgb_image, np.rad2deg(az), np.rad2deg(el), r)
        frames.append(annotated)

    imageio.mimwrite(out_path, frames, fps=10, quality=8)
    print(f"[✔] Vidéo rapprochement axe Z sauvegardée : {out_path}")
