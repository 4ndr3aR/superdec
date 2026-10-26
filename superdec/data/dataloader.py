import os
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset

from superdec.data.transform import RotateAroundAxis3d, Scale3d, RandomMove3d, Compose, rotate_around_axis
from superdec.data.xzm import load_array_archive, load_array_archive_cached
SHAPENET_CATEGORIES = {
    "04379243": "table", "02958343": "car", "03001627": "chair", "02691156": "airplane",
    "04256520": "sofa", "04090263": "rifle", "03636649": "lamp", "03691459": "loudspeaker",
    "02933112": "cabinet", "03211117": "display", "04401088": "telephone", "02828884": "bench",
    "04530566": "watercraft"
}

def normalize_points(points):
    translation = points.mean(0)
    points = points - translation
    scale = 2 * np.max(np.abs(points))
    points = points / scale
    return points, translation, scale

def denormalize_points(points, translation, scale, z_up=False):
    scale = scale[:,None,None]
    translation = translation[:, None, :]
    points = points * scale + translation
    if z_up:
        points = rotate_around_axis(points.cpu().numpy(), axis=(1,0,0), angle = np.pi/2, center_point=np.zeros(3))
        points = torch.from_numpy(points)
    return points

def denormalize_outdict(outdict, translation, scale, z_up=False):
    scale = scale[:,None,None]
    translation = translation[:, None, :]
    outdict['scale'] = outdict['scale'] * scale
    outdict['trans'] = outdict['trans'] * scale + translation 
    if z_up:
        # transform sq by updating translation and rotation
        outdict['trans'] = torch.tensor(rotate_around_axis(outdict['trans'].cpu().numpy(), axis=(1,0,0), angle = np.pi/2, center_point=np.zeros(3)))
        outdict['rotate'] = outdict['rotate'].cpu().numpy()
        rot_x_90 = np.array([[1,0,0],[0,0,-1],[0,1,0]])
        for i in range(outdict['rotate'].shape[0]):
            outdict['rotate'][i] = rot_x_90 @ outdict['rotate'][i]
        outdict['rotate'] = torch.from_numpy(outdict['rotate'])
        
    return outdict

def get_transforms(split: str, cfg):
    if split != 'train' or 'trainer' not in cfg or not cfg.trainer.augmentations:
        return None

    return Compose([
        Scale3d(),
        RotateAroundAxis3d(rotation_limit=np.pi / 24, axis=(0, 0, 1)),
        RotateAroundAxis3d(rotation_limit=np.pi / 24, axis=(1, 0, 0)),
        RotateAroundAxis3d(rotation_limit=np.pi, axis=(0, 1, 0)),
        RandomMove3d(
            x_min=-0.1, x_max=0.1,
            y_min=-0.05, y_max=0.05,
            z_min=-0.1, z_max=0.1
        ),
    ])

class ScenesDataset(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.gt = cfg.scenes_dataset.gt
        gt_suffix = "_gt" if self.gt else ""
        self.subfolder = f"pc{gt_suffix}"
        self.path = os.path.join(cfg.scenes_dataset.path)
        self.split = cfg.scenes_dataset.split
        self.z_up = True
        self.fps = cfg.scenes_dataset.fps if 'fps' in cfg.scenes_dataset else False
        self.scenes = self._load_scenes()
        self.models = self._gather_models()

    def _load_scenes(self):
        split_txt = f'{self.split}.txt'
        if not os.path.exists(os.path.join(self.path, split_txt)):
            print('Split %s does not exist.' % (split_txt))
        with open(os.path.join(self.path, split_txt), 'r') as f:
            scenes_names = f.read().split('\n')
        return scenes_names

    def _gather_models(self):
        models = []
        for s in self.scenes:
            try:
                scene_path = os.path.join(self.path, s, self.subfolder)
                model_ids = [os.path.splitext(f)[0] for f in os.listdir(scene_path) if f.endswith(".npz")]
                models.extend([{'scene': s, 'model_id': m} for m in model_ids])
            except FileNotFoundError:
                continue
        return models

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):
        model = self.models[idx]
        model_path = os.path.join(self.path, model['scene'], self.subfolder, f"{model['model_id']}.npz")
        
        pc_data = np.load(model_path)
        points_tmp = pc_data['points']

        n_points = points_tmp.shape[0]

        if n_points >= 4096:
            if self.fps:
                import open3d as o3d  # lazy: only the optional FPS path needs it
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_tmp)
                points = np.asarray(pcd.farthest_point_down_sample(4096).points)
            else:
                idxs = np.random.choice(n_points, 4096, replace=False)
                points = points_tmp[idxs]
        else:
            idxs = np.random.choice(n_points, 4096)
            points = points_tmp[idxs]

        if self.z_up:
            points = rotate_around_axis(points, axis=(1,0,0), angle = -np.pi/2, center_point=np.zeros(3))

        points, translation, scale  = normalize_points(points)

        return {
            "points": torch.from_numpy(points),
            "translation": torch.from_numpy(translation),
            "scale": scale,
            "z_up": self.z_up,
            "point_num": points.shape[0],
            "model_id": model
        }

    def name(self):
        return 'ScenesDataset'


class Scene(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.gt = cfg.scene.gt
        gt_suffix = "_gt" if self.gt else ""
        self.path = os.path.join(cfg.scene.path, cfg.scene.name, f"pc{gt_suffix}")
        self.z_up = cfg.scene.z_up 
        self._gather_models()

    def _gather_models(self):
        self.models = [os.path.splitext(f)[0] for f in os.listdir(self.path) if f.endswith(".npz")]

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):
        model = self.models[idx]
        
        pc_data = np.load(os.path.join(self.path, f"{model}.npz"))
        points_tmp = pc_data['points']

        n_points = points_tmp.shape[0]

        if n_points >= 4096:
            idxs = np.random.choice(n_points, 4096, replace=False)
            points = points_tmp[idxs]
        else:
            idxs = np.random.choice(n_points, 4096)
            points = points_tmp[idxs]

        if self.z_up:
            points = rotate_around_axis(points, axis=(1,0,0), angle = -np.pi/2, center_point=np.zeros(3))

        points, translation, scale  = normalize_points(points)

        return {
            "points": torch.from_numpy(points),
            "translation": torch.from_numpy(translation),
            "scale": scale,
            "z_up": self.z_up,
            "point_num": points.shape[0],
            "model_id": model
        }

    def name(self):
        return 'Scene'



class ShapeNet(Dataset):
    def __init__(self, split: str, cfg):
        super().__init__()
        self.split = split
        self.data_root = cfg.shapenet.path

        self.transform = get_transforms(split, cfg)
        self.normalize = cfg.shapenet.normalize

        self.categories = self._load_categories(cfg.shapenet.categories)
        self.models = self._gather_models()

    def _load_categories(self, categories):
        if categories is None:
            return [d for d in os.listdir(self.data_root)
                    if os.path.isdir(os.path.join(self.data_root, d))]
        print(f"Categories for split '{self.split}': {', '.join(SHAPENET_CATEGORIES[c] for c in categories)}")
        return categories

    def _gather_models(self):
        models = []
        for c in self.categories:
            category_path = os.path.join(self.data_root, c)
            split_file = os.path.join(category_path, f'{self.split}.lst')
            if not os.path.exists(split_file):
                continue
            with open(split_file, 'r') as f:
                model_ids = [line.strip() for line in f if line.strip()]
            models.extend([{'category': c, 'model_id': m} for m in model_ids])
        return models

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):
        model = self.models[idx]
        model_path = os.path.join(self.data_root, model['category'], model['model_id'])
        

        if self.split == 'test': 
            try : # for more rigorous evaluation on the test set, we use the 4096 points version downsampled with fps
                pc_data = np.load(os.path.join(model_path, "pointcloud_4096.npz"))
                points = pc_data["points"]
                normals = pc_data["normals"]
            except FileNotFoundError:
                pc_data = np.load(os.path.join(model_path, "pointcloud.npz"))
                n_points = pc_data["points"].shape[0]
                idxs = np.random.choice(n_points, 4096, replace=False)
                points = pc_data["points"][idxs]
                normals = pc_data["normals"][idxs]
            
        else:
            pc_data = np.load(os.path.join(model_path, "pointcloud.npz"))
            n_points = pc_data["points"].shape[0]
            idxs = np.random.choice(n_points, 4096, replace=False)
            points = pc_data["points"][idxs]
            normals = pc_data["normals"][idxs]

        if self.normalize:
            points, translation, scale  = normalize_points(points)
        else:
            translation = np.zeros(3)
            scale = 1.0

        if self.transform is not None:
            t_data = self.transform(points=points, normals=normals)
            points = t_data['points']
            normals = t_data['normals']

        return {
            "points": torch.from_numpy(points),
            "normals": torch.from_numpy(normals),
            "translation": torch.from_numpy(translation),
            "scale": scale,
            "point_num": points.shape[0],
            "model_id": model['model_id']
        }

    def name(self):
        return 'ShapeNet'


def knn_pca_normals(p, k=16):
    """Unit per-point surface normals for a (N,3) dense cloud: kNN + per-point PCA, smallest eigenvector,
    signed away from the cloud centroid. Rotation-covariant, so normals recomputed AFTER the point-cloud
    augmentation compose (any applied rotation is exactly reflected). N==0 -> (0,3)."""
    p = np.ascontiguousarray(p, dtype=np.float64)
    n = p.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    from scipy.spatial import cKDTree                       # pod venv (scipy 1.18); ~12ms at N=4096
    nbrs = cKDTree(p).query(p, k=min(k, n) + 1, workers=-1)[1]       # (N, k+1) incl. self
    rel = p[nbrs] - p[:, None, :]                              # (N, k+1, 3)
    cov = np.einsum("nki,nkj->nij", rel[:, 1:], rel[:, 1:])    # (N,3,3); the self term vanishes (rel=0)
    nrm = np.linalg.eigh(cov)[1][:, :, 0]                      # (N,3) smallest eigenvector = surface normal
    cen = nrm.mean(0)                                          # (3,) mean-normal vector for the outward sign
    d = (nrm * (p - cen)).sum(1, keepdims=True)
    d[d == 0.0] = 1.0
    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) * d)
    return np.ascontiguousarray(nrm, dtype=np.float32)


class ShapeNetXZ(Dataset):
    """Same ShapeNet contract (points/normals/translation/scale) fed from the fleet's .xz surface clouds:
    one sample per live (scene, obj_k); the 4096-point canonical surface block obj_k_surface_pts from each
    scene's vt3learn_cache/bundle_index.xz, unit-cube normalized exactly like ShapeNet (normalize_points) and
    augmented by the SAME stock train-time pipeline (get_transforms) so this run is regime-identical to the
    NPZ fleet run except for the normal source. Normals are the one thing the .xz does not carry: they are
    recomputed by knn_pca_normals on the FINAL (post-augmentation) cloud -- rotation-covariant, so the
    composed rotation/scale/translate lands on them exactly. Splits are the platform's scene lists
    (xzfleet_splits/{train,val,test}.txt -- the same 2320/270/140-scene split the platform trains on)."""

    def __init__(self, split, cfg):
        super().__init__()
        self.split = split
        self.fleet = str(cfg.shapenet.xz_fleet)
        if not os.path.isdir(self.fleet):
            raise AssertionError(f"ShapeNetXZ: fleet root missing {self.fleet}")
        self.n_knn = int(cfg.shapenet.xz_n_knn)
        self.normalize = cfg.shapenet.normalize
        self.transform = get_transforms(split, cfg)          # stock train-time pipeline (None off-train)
        self.models = self._scan(str(cfg.shapenet.xzfleet_splits_dir))

    def _scan(self, splits_dir):
        txt = os.path.join(splits_dir, f"{self.split}.txt")
        if not os.path.isfile(txt):
            raise FileNotFoundError(f"ShapeNetXZ: no split file {txt}")
        models, n_scenes = [], 0
        for line in open(txt):
            sc = line.strip()
            if not sc:
                continue
            n_scenes += 1
            bp = os.path.join(self.fleet, sc, "vt3learn_cache", "bundle_index.xz")
            d = load_array_archive(bp)                      # loud on a malformed/missing bundle
            for k in range(int(np.asarray(d["n_obj"]))):
                models.append((sc, k, f"obj_{k}_surface_pts"))
        assert models, f"ShapeNetXZ: split {self.split} has no live object rows"
        print(f"ShapeNetXZ[{self.split}]: {n_scenes} scenes -> {len(models)} object clouds (raw .xz, per-object regime)")
        return models

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):
        sc, k, sk = self.models[idx]
        d = load_array_archive_cached(os.path.join(self.fleet, sc, "vt3learn_cache", "bundle_index.xz"))
        pts = np.asarray(d[sk], dtype=np.float64)           # (4096,3) canonical meters, per-object scale ~0.32-0.5
        assert pts.shape == (4096, 3), f"{sc}/{sk}: {pts.shape} != (4096,3)"
        assert np.all(np.isfinite(pts)), f"{sc}/{sk}: non-finite surface rows"

        # Mirror the standalone ShapeNet contract EXACTLY: normalize_points only when cfg.normalize,
        # else raw raw points with zero translation / scale 1 (the voxel encoder's `normalize: True`
        # then rescales -- the fleet NPZ regime, which the checkpoint was trained in).
        if self.normalize:
            points, translation, scale = normalize_points(pts)
        else:
            points, translation, scale = pts, np.zeros(3), 1.0

        if self.transform is not None:
            # No fake pre-normals: pass a zero placeholder, apply only the POINT transforms, then compute
            # kNN-PCA normals on the FINAL (post-aug) cloud -- covariant, so the composed rotation/scale
            # lands on them exactly (the stock pipeline's `apply_to_normals` would otherwise rotate/
            # misalign any pre-baked normal against the augmented points).
            t = self.transform(points=points, normals=np.zeros_like(points))
            points = t["points"]
        normals = knn_pca_normals(points, k=self.n_knn)

        assert points.shape == normals.shape and points.shape[1] == 3
        return {
            "points": torch.from_numpy(np.ascontiguousarray(points, dtype=np.float32)),
            "normals": torch.from_numpy(normals),
            "translation": torch.from_numpy(np.asarray(translation, dtype=np.float32)),
            "scale": float(scale),
            "point_num": points.shape[0],
            "model_id": f"{sc}/obj_{k}",
        }

    def name(self):
        return "ShapeNetXZ"