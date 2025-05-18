import json
import numpy as np
from rpcfit.rpc_fit import calibrate_rpc
from rpcm.rpc_model import RPCModel

# === Charger le modèle RPC d'origine ===
with open("JAX_214_007_RGB.json", "r") as f:
    data = json.load(f)
rpc_orig = RPCModel(data["rpc"], dict_format="rpcm")

# === Définir le point central (au sol) autour duquel orbiter ===
target_lon = rpc_orig.lon_offset
target_lat = rpc_orig.lat_offset
target_alt = rpc_orig.alt_offset

# === Générer une grille de points 3D autour de la scène ===
num_xy = 10
num_z = 3
delta_lon = rpc_orig.lon_scale * 0.2
delta_lat = rpc_orig.lat_scale * 0.2
delta_alt = rpc_orig.alt_scale * 0.05

lon_vals = np.linspace(-delta_lon, delta_lon, num_xy) + target_lon
lat_vals = np.linspace(-delta_lat, delta_lat, num_xy) + target_lat
alt_vals = np.linspace(-delta_alt, delta_alt, num_z) + target_alt

lon, lat, alt = np.meshgrid(lon_vals, lat_vals, alt_vals)
scene_points_3d = np.stack([lon.ravel(), lat.ravel(), alt.ravel()], axis=1)

# === Paramètres d'orbite ===
num_poses = 8
radius_deg = 0.0003  # ≈ 30m
orbit_alt_offset = 20  # élévation supplémentaire si souhaitée
angles = np.linspace(0, 2 * np.pi, num_poses, endpoint=False)

# === Génération des poses ===
for i, theta in enumerate(angles):
    # Définir une nouvelle position caméra en orbite
    cam_lon = target_lon + radius_deg * np.cos(theta)
    cam_lat = target_lat + radius_deg * np.sin(theta)
    cam_alt = target_alt + orbit_alt_offset  # simulation d'une orbite élevée

    # Créer un faux modèle RPC à partir de la position orbite
    rpc_fake = RPCModel(data["rpc"], dict_format="rpcm")
    rpc_fake.lon_offset = cam_lon
    rpc_fake.lat_offset = cam_lat
    rpc_fake.alt_offset = cam_alt

    # Projeter les points 3D avec cette position
    points_2d = np.array([rpc_fake.projection(*p) for p in scene_points_3d])
    points_2d = np.flip(points_2d, axis=1)  # (col, row) → (row, col)

    # Calibrer un nouveau modèle RPC basé sur cette "vue"
    rpc_artificial = calibrate_rpc(
        target=points_2d,
        input_locs=scene_points_3d,
        separate=True,
        orientation="projection",
        init=rpc_fake  # on garde l'offset/scale basé sur la nouvelle position
    )

    # Sauvegarder ce modèle RPC artificiel
    output_rpc_dict = {
        "row_offset": rpc_artificial.row_offset,
        "col_offset": rpc_artificial.col_offset,
        "lat_offset": rpc_artificial.lat_offset,
        "lon_offset": rpc_artificial.lon_offset,
        "alt_offset": rpc_artificial.alt_offset,
        "row_scale": rpc_artificial.row_scale,
        "col_scale": rpc_artificial.col_scale,
        "lat_scale": rpc_artificial.lat_scale,
        "lon_scale": rpc_artificial.lon_scale,
        "alt_scale": rpc_artificial.alt_scale,
        "row_num": list(rpc_artificial.row_num),
        "row_den": list(rpc_artificial.row_den),
        "col_num": list(rpc_artificial.col_num),
        "col_den": list(rpc_artificial.col_den)
    }

    final_json = data.copy()
    final_json["rpc"] = output_rpc_dict
    output_name = f"JAX_214_007_RPC_orbite_pose_{i:02}.json"
    with open(output_name, "w") as f:
        json.dump(final_json, f, indent=2)

    print(f"✅ Pose orbitale {i} sauvegardée → {output_name}")
