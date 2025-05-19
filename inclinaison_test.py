import json
import numpy as np
from rpcfit.rpc_fit import calibrate_rpc
from rpcm.rpc_model import RPCModel

# === Charger le fichier RPC original ===
with open("JAX_214_007_RGB.json", "r") as f:
    data = json.load(f)

rpc_orig = RPCModel(data["rpc"], dict_format="rpcm")

# === Cible au sol (centre de la scène initiale) ===
target_lon = rpc_orig.lon_offset
target_lat = rpc_orig.lat_offset
target_alt = rpc_orig.alt_offset

# === Grille autour de la zone observée (paramètres) ===
num_xy, num_z = 10, 3
delta_lon = rpc_orig.lon_scale * 0.2
delta_lat = rpc_orig.lat_scale * 0.2
delta_alt = rpc_orig.alt_scale * 0.05

# === Paramètres d'inclinaison orbitale ===
num_poses = 8
inclination_radius_deg = 0.0004  # déplacement caméra lat/lon (~40m)
azimuths = np.linspace(0, 2 * np.pi, num_poses, endpoint=False)

for i, az in enumerate(azimuths):
    # === Déplacer la caméra autour de la cible selon l'azimut ===
    cam_lon = target_lon + inclination_radius_deg * np.cos(az)
    cam_lat = target_lat + inclination_radius_deg * np.sin(az)
    cam_alt = target_alt + 30  # élévation caméra

    # === Calculer la direction de visée
    dir_lon = target_lon - cam_lon
    dir_lat = target_lat - cam_lat

    # === Déplacer la grille pour qu'elle soit en face de la caméra
    scene_center_lon = target_lon + dir_lon
    scene_center_lat = target_lat + dir_lat
    scene_center_alt = target_alt

    # === Grille 3D alignée avec l'axe visuel
    lon_vals = np.linspace(-delta_lon, delta_lon, num_xy) + scene_center_lon
    lat_vals = np.linspace(-delta_lat, delta_lat, num_xy) + scene_center_lat
    alt_vals = np.linspace(-delta_alt, delta_alt, num_z) + scene_center_alt

    lon, lat, alt = np.meshgrid(lon_vals, lat_vals, alt_vals)
    scene_points_3d = np.stack([lon.ravel(), lat.ravel(), alt.ravel()], axis=1)

    # === Créer un faux modèle RPC depuis cette position
    rpc_fake = RPCModel(data["rpc"], dict_format="rpcm")
    rpc_fake.lon_offset = cam_lon
    rpc_fake.lat_offset = cam_lat
    rpc_fake.alt_offset = cam_alt

    # === Projeter les points avec le modèle déplacé
    points_2d = np.array([rpc_fake.projection(*p) for p in scene_points_3d])
    points_2d = np.flip(points_2d, axis=1)  # (col, row) → (row, col)

    # === Recalibrer un nouveau modèle RPC
    rpc_inclined = calibrate_rpc(
        target=points_2d,
        input_locs=scene_points_3d,
        separate=True,
        orientation="projection",
        init=rpc_fake
    )

    # === Sauvegarder le modèle RPC recalibré
    rpc_dict = {
        "row_offset": rpc_inclined.row_offset,
        "col_offset": rpc_inclined.col_offset,
        "lat_offset": rpc_inclined.lat_offset,
        "lon_offset": rpc_inclined.lon_offset,
        "alt_offset": rpc_inclined.alt_offset,
        "row_scale": rpc_inclined.row_scale,
        "col_scale": rpc_inclined.col_scale,
        "lat_scale": rpc_inclined.lat_scale,
        "lon_scale": rpc_inclined.lon_scale,
        "alt_scale": rpc_inclined.alt_scale,
        "row_num": list(rpc_inclined.row_num),
        "row_den": list(rpc_inclined.row_den),
        "col_num": list(rpc_inclined.col_num),
        "col_den": list(rpc_inclined.col_den)
    }

    final_json = data.copy()
    final_json["rpc"] = rpc_dict
    output_name = f"JAX_214_007_RPC_inclinaison_{i:02}.json"

    with open(output_name, "w") as f:
        json.dump(final_json, f, indent=2)

    print(f"✅ Pose inclinée #{i} sauvegardée → {output_name}")
