import cv2
import numpy as np
import json
import os
from rpcm.rpc_model import RPCModel
from rpcm.stereo_model import StereoModel
from rpcfit.rpc_fit import calibrate_rpc

# === PARAMÈTRES ===
img1_path = "JAX_214_007_RGB.tif"
img2_path = "JAX_214_008_RGB.tif"
rpc1_json = "JAX_214_007_RGB.json"
rpc2_json = "JAX_214_008_RGB.json"
output_prefix = "JAX_214_007_RPC"
nb_poses = 10
focal = 30000

# === 1. Charger les images ===
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
h, w = img1.shape

# === 2. Charger les RPC ===
with open(rpc1_json) as f:
    rpc1_data = json.load(f)
rpc1 = RPCModel(rpc1_data["rpc"], dict_format="rpcm")

with open(rpc2_json) as f:
    rpc2 = RPCModel(json.load(f)["rpc"], dict_format="rpcm")

# === 3. Détection et appariement de points ===
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        pt1 = kp1[m.queryIdx].pt
        pt2 = kp2[m.trainIdx].pt
        good_matches.append((pt1, pt2))

if len(good_matches) < 20:
    raise RuntimeError("Pas assez de correspondances valides")

pts1 = np.array([pt1 for pt1, _ in good_matches])
pts2 = np.array([pt2 for _, pt2 in good_matches])

# === 4. Triangulation avec RPC ===
z0 = 100.0
alts_guess = np.full(len(pts1), z0)
stereo = StereoModel(rpc1, rpc2)
lons, lats, alts = stereo.inverse_stereo(
    col1=pts1[:, 0], row1=pts1[:, 1],
    col2=pts2[:, 0], row2=pts2[:, 1],
    z0=alts_guess
)
points_3d = np.vstack([lons, lats, alts]).T

# === 5. Estimation de la pose pinhole ===
K = np.array([[focal, 0, w / 2],
              [0, focal, h / 2],
              [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))
success, rvec, tvec = cv2.solvePnP(points_3d, pts1, K, dist_coeffs)
if not success:
    raise RuntimeError("solvePnP a échoué")

R, _ = cv2.Rodrigues(rvec)
C = -R.T @ tvec

# === 6. Génération et sauvegarde des poses RPC simulées ===
for i in range(nb_poses):
    angle = np.random.uniform(-5, 5)
    dz = np.random.uniform(-50, 50)

    delta_rvec = np.array([0, 0, np.deg2rad(angle)], dtype=np.float32)
    R_delta, _ = cv2.Rodrigues(delta_rvec)
    R_new = R_delta @ R
    C_new = C + np.array([[0], [0], [dz]])
    t_new = -R_new @ C_new

    projected, _ = cv2.projectPoints(points_3d, cv2.Rodrigues(R_new)[0], t_new, K, dist_coeffs)
    pts_proj = projected.squeeze()

    cam_coords = (R_new @ (points_3d.T - C_new)).T
    valid = cam_coords[:, 2] > 0
    pts2d_valid = pts_proj[valid]
    pts3d_valid = points_3d[valid]

    if len(pts2d_valid) < 20:
        print(f"Pose {i} ignorée (pas assez de points valides)")
        continue

    rpc_new = calibrate_rpc(target=pts2d_valid, input_locs=pts3d_valid)

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

    final_json = rpc1_data.copy()
    final_json["rpc"] = rpc_dict

    out_path = f"{output_prefix}_{i:02d}.json"
    with open(out_path, "w") as f:
        json.dump(final_json, f, indent=2)

    print(f"Pose simulée {i} sauvegardée dans {out_path}")
