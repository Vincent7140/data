from datasets import SatelliteRGBDEPDataset
dataset = SatelliteRGBDEPDataset(...)
sample = dataset[0]
rays_ref = sample["rays"]



rays_o = rays_ref[:, 0:3]
rays_d = rays_ref[:, 3:6]
center = rays_o.mean(dim=0)  # position du satellite
normal = rays_d.mean(dim=0)  # direction moyenne (environ vers le sol)

import math
N_views = 20  # nombre de vues sur l’orbite
radius = 1000  # distance satellite-scène
up = torch.tensor([0.0, 0.0, 1.0])  # Z vertical

poses = []
for i in range(N_views):
    angle = 2 * math.pi * i / N_views
    offset = torch.tensor([math.cos(angle), math.sin(angle), 0.0]) * radius
    cam_origin = center + offset
    cam_dir = (center - cam_origin)
    cam_dir = cam_dir / torch.norm(cam_dir)
    poses.append((cam_origin, cam_dir))


W = H = 512
span = 500  # zone couverte au sol
x = torch.linspace(-span, span, W)
y = torch.linspace(-span, span, H)
gx, gy = torch.meshgrid(x, y, indexing='xy')

rays_all = []
for origin, direction in poses:
    rays_o = torch.stack([
        origin[0] + gx,
        origin[1] + gy,
        origin[2].repeat(gx.shape)
    ], dim=-1).reshape(-1, 3)

    rays_d = direction.repeat(rays_o.shape[0], 1)
    nears = torch.zeros((rays_o.shape[0], 1))
    fars = torch.ones((rays_o.shape[0], 1)) * 2000
    sun_dir = torch.tensor([[1.0, 0.0, 1.0]]).repeat(rays_o.shape[0], 1)

    rays = torch.cat([rays_o, rays_d, nears, fars, sun_dir], dim=1)
    rays_all.append(rays)



for i, rays in enumerate(rays_all):
    results = batched_inference(models, rays.cuda(), ts=None, args=args)
    save_image_from_results(results, out_path=f"orbit_view_{i:02d}.png", ...)
