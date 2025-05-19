import numpy as np
import json
from pyproj import Transformer
from rpcfit.rpc_fit import calibrate_rpc
from rpcm.rpc_model import RPCModel

# === Paramètres ===
json_input = "JAX_214_007_RGB.json"
output_prefix = "JAX_214_007_RPC_orbite"
num_poses = 8
orbit_radius_m = 40  # en mètres (≈ décalage lat/lon pour ~obliquité)
alt_delta = 30  # mètres d'altitude satellite en plus
grid_size_xy, grid_size_z = 10, 3

# === Charger RPC d’origine
with open(json_input, "r") as f:
    data = json.load(f)
rpc_orig = RPCModel(data["rpc"], dict_format="rpcm")

# === Convertir le centre de la scène en ECEF
center_lat, center_lon, center_alt = rpc_orig.lat_offset, rpc_orig.lon_offset, rpc_orig.alt_offset
transformer_to_ecef = Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)
x_c, y_c, z_c = transformer_to_ecef.transform(center_lon, center_lat, center_alt)

# === Grille de points 3D autour de la scène
delta_lon = rpc_orig.lon_scale * 0.2
delta_lat = rpc_orig.lat_scale * 0.2
delta_alt = rpc_orig.alt_scale * 0.05

lon_vals = np.linspace(-delta_lon, delta_lon, grid_size_xy) + center_lon
lat_vals = np.linspace(-delta_lat, delta_lat, grid_size_xy) + center_lat
alt_vals = np.linspace(-delta_alt, delta_alt, grid_size_z) + center_alt

lon, lat, alt = np.meshgrid(lon_vals, lat_vals, alt_vals)
scene_points_3d = np.stack([lon.ravel(), lat.ravel(), alt.ravel()], axis=1)

# === Préparer transformateur pour revenir en lat/lon
transformer_from_ecef = Transformer.from_crs("epsg:4978", "epsg:4326", always_xy=True)

# === Générer plusieurs poses autour de la scène
azimuths = np.linspace(0, 2 * np.pi, num_poses, endpoint=False)

for i, az in enumerate(azimuths):
    # Position satellite en orbite (ECEF)
    dx = orbit_radius_m * np.cos(az)
    dy = orbit_radius_m * np.sin(az)
    dz = alt_delta

    sat_x, sat_y, sat_z = x_c + dx, y_c + dy, z_c + dz

    # Vecteurs directionnels de la caméra
    z_axis = np.array([x_c - sat_x, y_c - sat_y, z_c - sat_z])
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross([0, 0, 1], z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    # Projeter les points 3D sur un plan image fictif
    points_ecef = np.array(transformer_to_ecef.transform(scene_points_3d[:, 0], scene_points_3d[:, 1], scene_points_3d[:, 2])).T
    vecs = points_ecef - np.array([sat_x, sat_y, sat_z])
    proj_x = np.dot(vecs, x_axis)
    proj_y = np.dot(vecs, y_axis)

    # Normaliser les coordonnées image (en pixels fictifs)
    col_scale, row_scale = 5000, 5000
    cols = proj_x * col_scale + 4000
    rows = proj_y * row_scale + 4000
    points_2d = np.stack([rows, cols], axis=1)

    # === Calibrer un nouveau RPC
    rpc_new = calibrate_rpc(
        target=points_2d,
        input_locs=scene_points_3d,
        separate=True,
        orientation="projection"
    )

    # === Sauvegarder le modèle RPC synthétique
    rpc_dict = {
        "row_offset": rpc_new.row_offset,
        "col_offset": rpc_new.col_offset,
        "lat_offset": rpc_new.lat_offset,
        "lon_offset": rpc_new.lon_offset,
        "alt_offset": rpc_new.alt_offset,
        "row_scale": rpc_new.row_scale,
        "col_scale": rpc_new.col_scale,
        "lat_scale": rpc_new.lat_scale,
        "lon_scale": rpc_new.lon_scale,
        "alt_scale": rpc_new.alt_scale,
        "row_num": list(rpc_new.row_num),
        "row_den": list(rpc_new.row_den),
        "col_num": list(rpc_new.col_num),
        "col_den": list(rpc_new.col_den)
    }

    json_out = data.copy()
    json_out["rpc"] = rpc_dict
    json_out["img"] = "JAX_214_007_RGB_fake.tif"  # image factice
    out_path = f"{output_prefix}_{i:02d}.json"

    with open(out_path, "w") as f:
        json.dump(json_out, f, indent=2)

    print(f"✅ Pose synthétique #{i} sauvegardée → {out_path}")
