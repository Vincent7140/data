import json
import numpy as np
from rpcfit.rpc_fit import calibrate_rpc
from rpcm.rpc_model import RPCModel

# === Charger le fichier RPC original ===
with open("JAX_214_007_RGB.json", "r") as f:
    data = json.load(f)

rpc_orig = RPCModel(data["rpc"], dict_format="rpcm")

# === Cible fixe au sol ===
target_lon = rpc_orig.lon_offset
target_lat = rpc_orig.lat_offset
target_alt = rpc_orig.alt_offset

# === Générer une grille de points 3D autour de la cible ===
num_xy, num_z = 10, 3
delta_lon = rpc_orig.lon_scale * 0.2
delta_lat = rpc_orig.lat_scale * 0.2
delta_alt = rpc_orig.alt_scale * 0.05

lon_vals = np.linspace(-delta_lon, delta_lon, num_xy) + target_lon
lat_vals = np.linspace(-delta_lat, delta_lat, num_xy) + target_lat
alt_vals = np.linspace(-delta_alt, delta_alt, num_z) + target_alt

lon, lat, alt = np.meshgrid(lon_vals, lat_vals, alt_vals)
scene_points_3d = np.stack([lon.ravel(), lat.ravel(), alt.ravel()], axis=1)

# === Paramètres de variation d'azimut (vue inclinée) ===
num_poses = 8
inclination_radius_deg = 0.0004  # déplacement pour simuler une inclinaison (~40m)
azimuths = np.linspace(0, 2 * np.pi, num_poses, endpoint=False)

for i, az in enumerate(azimuths):
    # Simuler une prise de vue inclinée depuis une position lat/lon décalée
    cam_lon = target_lon + inclination_radius_deg * np.cos(az)
    cam_lat = target_lat + inclination_radius_deg * np.sin(az)
    cam_alt = target_alt + 30  # légèrement en hauteur

    # Créer un modèle RPC "faux" depuis cette position
    rpc_fake = RPCModel(data["rpc"], dict_format="rpcm")
    rpc_fake.lon_offset = cam_lon
    rpc_fake.lat_offset = cam_lat
    rpc_fake.alt_offset = cam_alt

    # Projeter les points depuis cette vue inclinée
    points_2d = np.array([rpc_fake.projection(*p) for p in scene_points_3d])
    points_2d = np.flip(points_2d, axis=1)  # (col, row) → (row, col)

    # Calibrer un nouveau modèle RPC cohérent avec cette vue
    rpc_inclined = calibrate_rpc(
        target=points_2d,
        input_locs=scene_points_3d,
        separate=True,
        orientation="projection",
        init=rpc_fake
    )

    # Sauvegarder le modèle RPC incliné
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
