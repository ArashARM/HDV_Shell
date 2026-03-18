import numpy as np
import pandas as pd
import torch

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IGESControl import IGESControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_IN, TopAbs_ON
from OCC.Core.TopoDS import topods
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone
from OCC.Core.BRepTools import breptools
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.Poly import Poly_Triangle
from OCC.Core.gp import gp_Pnt, gp_Pnt2d
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.BRepClass import BRepClass_FaceClassifier
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box
from scipy.spatial import Delaunay

class CADTensorGenerator:
    """
    Load a CAD file, triangulate faces, compute differential geometry,
    and generate torch tensors for downstream ML / optimization pipelines.
    """

    def __init__(
        self,
        deflection: float = 0.5,
        angle: float = 0.5,
        metric_tol: float = 1e-9,
        det_min: float = 1e-5,
        n_u: int = 80,
        n_v: int = 40,
        device: str = "cpu",
    ):
        self.deflection = float(deflection)
        self.angle = float(angle)
        self.metric_tol = float(metric_tol)
        self.det_min = float(det_min)
        self.n_u = int(n_u)
        self.n_v = int(n_v)
        self.device = device

    # =========================================================================
    # 1) Load + face helpers
    # =========================================================================

    @staticmethod
    def load_shape(path: str):
        """Load STEP/IGES into a TopoDS_Shape."""
        p = path.lower()

        if p.endswith((".step", ".stp")):
            r = STEPControl_Reader()
            if r.ReadFile(path) != IFSelect_RetDone:
                raise RuntimeError("STEP read failed")
            r.TransferRoots()
            return r.OneShape()

        if p.endswith((".iges", ".igs")):
            r = IGESControl_Reader()
            if r.ReadFile(path) != IFSelect_RetDone:
                raise RuntimeError("IGES read failed")
            r.TransferRoots()
            return r.OneShape()

        raise ValueError("Unsupported file type (need .step/.stp/.iges/.igs)")

    @staticmethod
    def iter_faces(shape):
        """Yield TopoDS_Face items."""
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            yield topods.Face(exp.Current())
            exp.Next()

    @staticmethod
    def face_surface_type(face):
        """Return GeomAbs_* surface type."""
        adap = BRepAdaptor_Surface(face, True)
        return adap.GetType()

    @staticmethod
    def face_uv_periodicity(face):
        """
        Periodicity of the UNDERLYING surface (not trimmed face).
        """
        surf_ad = BRepAdaptor_Surface(face, False)
        u_per = bool(surf_ad.IsUPeriodic())
        v_per = bool(surf_ad.IsVPeriodic())
        u_period = float(surf_ad.UPeriod()) if u_per else None
        v_period = float(surf_ad.VPeriod()) if v_per else None
        return u_per, v_per, u_period, v_period

    @classmethod
    def is_manual_mesh_face(cls, face):
        """Plane/cylinder/cone use UV grid; others use OCC fallback."""
        return cls.face_surface_type(face) in (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone)

    @staticmethod
    def face_uv_bounds_and_surface(face):
        """Return (umin, umax, vmin, vmax) and underlying surface."""
        umin, umax, vmin, vmax = breptools.UVBounds(face)
        surf = BRep_Tool.Surface(face)
        return (float(umin), float(umax), float(vmin), float(vmax)), surf

    # =========================================================================
    # 2) Ensure OCC triangulation exists
    # =========================================================================

    @staticmethod
    def mesh_shape_force(shape, deflection=0.5, angle=0.5):
        """
        Ensure OCC has triangulation for all faces.
        """
        breptools.Clean(shape)

        mesher = None
        for args in [
            (shape, float(deflection)),
            (shape, float(deflection), False, float(angle), True),
            (shape, float(deflection), True, float(angle), True),
        ]:
            try:
                mesher = BRepMesh_IncrementalMesh(*args)
                break
            except TypeError:
                continue

        if mesher is None:
            raise RuntimeError("Could not construct BRepMesh_IncrementalMesh with known signatures.")

        if hasattr(mesher, "Perform"):
            mesher.Perform()

        return mesher

    # =========================================================================
    # 3) OCC triangulation reader
    # =========================================================================

    @staticmethod
    def triangulate_face_occ(face):
        """
        Read triangulation attached to 'face'.
        Returns: (V_xyz, UV_raw, F_idx) or None
        """
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is None:
            return None

        trsf = loc.Transformation()

        def _get(obj, name):
            if not hasattr(obj, name):
                return None
            a = getattr(obj, name)
            return a() if callable(a) else a

        V = []
        nodes_arr = _get(tri, "Nodes")
        if nodes_arr is not None:
            for i in range(1, nodes_arr.Size() + 1):
                p = nodes_arr.Value(i).Transformed(trsf)
                V.append([p.X(), p.Y(), p.Z()])
            nb_nodes = len(V)
        else:
            nb_nodes = _get(tri, "NbNodes")
            if nb_nodes is None or not hasattr(tri, "Node"):
                return None
            nb_nodes = int(nb_nodes)
            for i in range(1, nb_nodes + 1):
                try:
                    p = tri.Node(i).Transformed(trsf)
                except TypeError:
                    p = gp_Pnt()
                    tri.Node(i, p)
                    p = p.Transformed(trsf)
                V.append([p.X(), p.Y(), p.Z()])

        UV = [[None, None] for _ in range(nb_nodes)]
        uv_arr = _get(tri, "UVNodes")
        if uv_arr is not None:
            for i in range(1, uv_arr.Size() + 1):
                uv = uv_arr.Value(i)
                UV[i - 1] = [uv.X(), uv.Y()]
        else:
            if hasattr(tri, "UVNode"):
                for i in range(1, nb_nodes + 1):
                    try:
                        uv = tri.UVNode(i)
                    except TypeError:
                        uv = gp_Pnt2d()
                        tri.UVNode(i, uv)
                    UV[i - 1] = [uv.X(), uv.Y()]

        F = []
        tris_arr = _get(tri, "Triangles")
        if tris_arr is not None:
            for i in range(1, tris_arr.Size() + 1):
                a, b, c = tris_arr.Value(i).Get()
                F.append([a - 1, b - 1, c - 1])
        else:
            nb_tris = _get(tri, "NbTriangles")
            if nb_tris is None or not hasattr(tri, "Triangle"):
                return None
            nb_tris = int(nb_tris)
            for i in range(1, nb_tris + 1):
                try:
                    t = tri.Triangle(i)
                    a, b, c = t.Get()
                except TypeError:
                    t = Poly_Triangle()
                    tri.Triangle(i, t)
                    a, b, c = t.Get()
                F.append([a - 1, b - 1, c - 1])

        return np.asarray(V, float), np.asarray(UV, object), np.asarray(F, np.int64)

    # =========================================================================
    # 4) UV normalization + metric
    # =========================================================================

    @staticmethod
    def uv_raw_to_norm(u, v, umin, umax, vmin, vmax):
        Lu = float(umax - umin)
        Lv = float(vmax - vmin)
        if abs(Lu) < 1e-30:
            Lu = 1.0
        if abs(Lv) < 1e-30:
            Lv = 1.0
        return (float(u) - float(umin)) / Lu, (float(v) - float(vmin)) / Lv

    @staticmethod
    def metric_EFG_raw(surf, u, v, tol=1e-9):
        props = GeomLProp_SLProps(surf, float(u), float(v), 1, float(tol))
        if not props.IsNormalDefined():
            return None
        Su = props.D1U()
        Sv = props.D1V()
        E = Su.Dot(Su)
        F = Su.Dot(Sv)
        G = Sv.Dot(Sv)
        Su_xyz = (Su.X(), Su.Y(), Su.Z())
        Sv_xyz = (Sv.X(), Sv.Y(), Sv.Z())
        return Su_xyz, Sv_xyz, float(E), float(F), float(G)

    @staticmethod
    def metric_EFG_normalized(E, F, G, umin, umax, vmin, vmax):
        Lu = float(umax - umin)
        Lv = float(vmax - vmin)
        return float(E) * (Lu * Lu), float(F) * (Lu * Lv), float(G) * (Lv * Lv)

    # =========================================================================
    # 5) Manual UV-grid triangulation
    # =========================================================================

    @staticmethod
    def _classify_inside(face, u, v, classifier, tol):
        classifier.Perform(face, gp_Pnt2d(float(u), float(v)), float(tol))
        st = classifier.State()
        return (st == TopAbs_IN) or (st == TopAbs_ON)

    @classmethod
    def _should_wrap_u_true_seam(
        cls,
        face,
        surf,
        umin,
        umax,
        vmin,
        vmax,
        classifier=None,
        tol_uv=1e-7,
        n_test=9,
        xyz_tol=1e-6,
        period_rel_tol=1e-4,
    ):
        u_per, _, u_period, _ = cls.face_uv_periodicity(face)

        if not u_per:
            return False
        if u_period is None or u_period <= 0:
            return False

        u_span = float(umax - umin)
        if abs(u_span - u_period) > max(period_rel_tol * u_period, 1e-9):
            return False

        vs = np.linspace(vmin, vmax, int(max(3, n_test)))
        ok = 0
        tested = 0

        for vv in vs:
            try:
                p1 = surf.Value(float(umin), float(vv))
                p2 = surf.Value(float(umax), float(vv))
            except Exception:
                continue

            tested += 1
            dx = p1.X() - p2.X()
            dy = p1.Y() - p2.Y()
            dz = p1.Z() - p2.Z()

            if (dx * dx + dy * dy + dz * dz) <= float(xyz_tol) ** 2:
                ok += 1

        if tested == 0:
            return False

        return ok >= max(2, int(0.7 * tested))
    @classmethod
    def triangulate_face_uv_grid(cls, face, n_u=80, n_v=40, tol=1e-7):
        """
        Trim-aware UV grid triangulation for plane/cylinder/cone.
        """
        (umin, umax, vmin, vmax), surf = cls.face_uv_bounds_and_surface(face)
        stype = cls.face_surface_type(face)
        if stype not in (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone):
            return None

        n_u = int(max(2, n_u))
        n_v = int(max(2, n_v))
        tol = float(max(tol, 1e-12))

        classifier = BRepClass_FaceClassifier()

        wrap_u = cls._should_wrap_u_true_seam(
            face, surf, umin, umax, vmin, vmax, classifier, tol_uv=tol
        )

        vs = np.linspace(vmin, vmax, n_v + 1)
        us = np.linspace(umin, umax, n_u, endpoint=False) if wrap_u else np.linspace(umin, umax, n_u + 1)
        u_cols = len(us)

        V, UV = [], []
        idx = [[-1] * u_cols for _ in range(n_v + 1)]

        for j, v in enumerate(vs):
            for i, u in enumerate(us):
                if not cls._classify_inside(face, u, v, classifier, tol):
                    continue
                p = surf.Value(float(u), float(v))
                idx[j][i] = len(V)
                V.append([p.X(), p.Y(), p.Z()])
                UV.append([float(u), float(v)])

        if len(V) < 3:
            return None

        F = []
        for j in range(n_v):
            for i in range(u_cols):
                i2 = (i + 1) % u_cols if wrap_u else (i + 1)
                if (not wrap_u) and i2 >= u_cols:
                    continue
                a = idx[j][i]
                b = idx[j][i2]
                c = idx[j + 1][i2]
                d = idx[j + 1][i]
                if a < 0 or b < 0 or c < 0 or d < 0:
                    continue
                F.append([a, b, c])
                F.append([a, c, d])

        if len(F) == 0:
            return None

        return np.asarray(V, float), np.asarray(UV, float), np.asarray(F, np.int64)

    @classmethod
    def triangulate_face(cls, face, n_u=80, n_v=40, tol=1e-7):
        if cls.is_manual_mesh_face(face):
            tri = cls.triangulate_face_uv_grid(face, n_u=n_u, n_v=n_v, tol=tol)
            if tri is not None:
                return tri
        return cls.triangulate_face_occ(face)


    @staticmethod
    def _jacobian_area_density(surf, u: float, v: float, tol: float = 1e-9):
        props = GeomLProp_SLProps(surf, float(u), float(v), 1, float(tol))
        if not props.IsNormalDefined():
            return None
        Xu = props.D1U()
        Xv = props.D1V()
        cx = Xu.Crossed(Xv)
        J = float(cx.Magnitude())
        if not np.isfinite(J) or J <= 1e-14:
            return None
        return {
            "J": J,
            "Xu": np.array([Xu.X(), Xu.Y(), Xu.Z()], dtype=float),
            "Xv": np.array([Xv.X(), Xv.Y(), Xv.Z()], dtype=float),
        }

    @classmethod
    def sample_surface_points_uniform_weighted_pool(
        cls,
        face,
        M: int,
        pool_size: int = 20000,
        tol: float = 1e-7,
        metric_tol: float = 1e-9,
        use_fps: bool = True,
        fps_pool_factor: int = 4,
        device: str = "cpu",
        rng: np.random.Generator | None = None,
    ):
        if rng is None:
            rng = np.random.default_rng()

        M = int(M)
        pool_size = max(int(pool_size), M)

        (umin, umax, vmin, vmax), surf = cls.face_uv_bounds_and_surface(face)
        classifier = BRepClass_FaceClassifier()
        u_periodic, v_periodic, u_period, v_period = cls.face_uv_periodicity(face)

        wrap_u = False
        try:
            wrap_u = bool(
                cls._should_wrap_u_true_seam(
                    face, surf, float(umin), float(umax), float(vmin), float(vmax),
                    classifier=classifier, tol_uv=tol,
                )
            )
        except Exception:
            wrap_u = False

        uv_list, xyz_list, Xu_list, Xv_list, J_list = [], [], [], [], []
        trials = 0
        max_trials = max(20 * pool_size, 5000)

        while len(uv_list) < pool_size and trials < max_trials:
            trials += 1
            u = rng.uniform(float(umin), float(umax))
            v = rng.uniform(float(vmin), float(vmax))
            if not cls._classify_inside(face, u, v, classifier, tol):
                continue
            geom = cls._jacobian_area_density(surf, u, v, tol=metric_tol)
            if geom is None:
                continue
            p = surf.Value(float(u), float(v))
            uv_list.append([u, v])
            xyz_list.append([p.X(), p.Y(), p.Z()])
            Xu_list.append(geom["Xu"])
            Xv_list.append(geom["Xv"])
            J_list.append(geom["J"])

        if len(uv_list) < M:
            raise RuntimeError(f"Only got {len(uv_list)} valid candidates, need at least {M}.")

        uv_raw = np.asarray(uv_list, dtype=np.float32)
        xyz = np.asarray(xyz_list, dtype=np.float32)
        Xu = np.asarray(Xu_list, dtype=np.float32)
        Xv = np.asarray(Xv_list, dtype=np.float32)
        J = np.asarray(J_list, dtype=np.float64)

        probs = J / max(J.sum(), 1e-30)
        keep = min(len(uv_raw), max(M, fps_pool_factor * M))
        idx = rng.choice(len(uv_raw), size=keep, replace=False, p=probs)

        uv_raw = uv_raw[idx]
        xyz = xyz[idx]
        Xu = Xu[idx]
        Xv = Xv[idx]
        J = J[idx].astype(np.float32)

        if keep > M:
            if use_fps:
                pts_t = torch.tensor(xyz, dtype=torch.float32, device=device)
                idx_sel = cls.fps_3d(pts_t, S=M).detach().cpu().numpy()
            else:
                idx_sel = rng.choice(keep, size=M, replace=False)
            uv_raw = uv_raw[idx_sel]
            xyz = xyz[idx_sel]
            Xu = Xu[idx_sel]
            Xv = Xv[idx_sel]
            J = J[idx_sel]

        Lu = float(umax - umin) if abs(float(umax - umin)) > 1e-30 else 1.0
        Lv = float(vmax - vmin) if abs(float(vmax - vmin)) > 1e-30 else 1.0
        uv = np.empty_like(uv_raw, dtype=np.float32)
        uv[:, 0] = (uv_raw[:, 0] - float(umin)) / Lu
        uv[:, 1] = (uv_raw[:, 1] - float(vmin)) / Lv

        return {
            "uv_raw": uv_raw,
            "uv": uv,
            "points_xyz": xyz,
            "Xu": Xu,
            "Xv": Xv,
            "J": J,
            "u_raw_bounds": (float(umin), float(umax)),
            "v_raw_bounds": (float(vmin), float(vmax)),
            "u_periodic": bool(wrap_u),
            "v_periodic": bool(v_periodic),
            "u_period": None if u_period is None else float(u_period),
            "v_period": None if v_period is None else float(v_period),
        }

    @classmethod
    def triangulate_sampled_face(cls, face, uv_raw: np.ndarray, seam_rel_gap: float = 0.35, tol: float = 1e-7):
        if uv_raw.shape[0] < 3:
            return np.empty((0, 3), dtype=np.int64)

        (umin, umax, vmin, vmax), _ = cls.face_uv_bounds_and_surface(face)
        classifier = BRepClass_FaceClassifier()
        Lu = float(umax - umin) if abs(float(umax - umin)) > 1e-30 else 1.0
        Lv = float(vmax - vmin) if abs(float(vmax - vmin)) > 1e-30 else 1.0
        uv_norm = np.empty_like(uv_raw, dtype=np.float64)
        uv_norm[:, 0] = (uv_raw[:, 0] - float(umin)) / Lu
        uv_norm[:, 1] = (uv_raw[:, 1] - float(vmin)) / Lv

        try:
            tri = Delaunay(uv_norm)
        except Exception:
            return np.empty((0, 3), dtype=np.int64)

        simplices = np.asarray(tri.simplices, dtype=np.int64)
        if simplices.size == 0:
            return np.empty((0, 3), dtype=np.int64)

        keep = []
        for a, b, c in simplices:
            tri_uv_raw = uv_raw[[a, b, c]]
            ctr = tri_uv_raw.mean(axis=0)
            if not cls._classify_inside(face, float(ctr[0]), float(ctr[1]), classifier, tol):
                continue

            tri_uv_norm = uv_norm[[a, b, c]]
            edges = [
                np.linalg.norm(tri_uv_norm[0] - tri_uv_norm[1]),
                np.linalg.norm(tri_uv_norm[1] - tri_uv_norm[2]),
                np.linalg.norm(tri_uv_norm[2] - tri_uv_norm[0]),
            ]
            if max(edges) > seam_rel_gap:
                continue

            area2 = abs(np.cross(tri_uv_norm[1] - tri_uv_norm[0], tri_uv_norm[2] - tri_uv_norm[0]))
            if not np.isfinite(area2) or area2 <= 1e-12:
                continue
            keep.append([a, b, c])

        if len(keep) == 0:
            return np.empty((0, 3), dtype=np.int64)
        return np.asarray(keep, dtype=np.int64)

    def export_face_mesh_metric_points_triangulated(
        self,
        shape_path: str,
        M_per_face: int,
        pool_size_factor: int = 10,
        fps_pool_factor: int = 4,
        use_fps: bool = True,
        triangulation_max_edge_rel: float = 0.35,
    ):
        shape = self.load_shape(shape_path)

        mesh_rows = []
        face_rows = []

        for face_id, face in enumerate(self.iter_faces(shape)):
            try:
                samp = self.sample_surface_points_uniform_weighted_pool(
                    face=face,
                    M=M_per_face,
                    pool_size=max(int(pool_size_factor * M_per_face), M_per_face),
                    tol=1e-7,
                    metric_tol=self.metric_tol,
                    use_fps=use_fps,
                    fps_pool_factor=fps_pool_factor,
                    device=self.device,
                )
            except Exception as e:
                print(f"[warn] face {face_id}: point sampling failed: {e}")
                continue

            tri_faces = self.triangulate_sampled_face(
                face,
                samp["uv_raw"],
                seam_rel_gap=triangulation_max_edge_rel,
                tol=1e-7,
            )
            if tri_faces.shape[0] == 0:
                print(f"[warn] face {face_id}: no valid triangulation from sampled points, skipped")
                continue

            bbox = self.face_bounding_box(face)
            base_gvid = len(mesh_rows)
            for lvid in range(samp["uv_raw"].shape[0]):
                u_raw = float(samp["uv_raw"][lvid, 0])
                v_raw = float(samp["uv_raw"][lvid, 1])
                x, y, z = map(float, samp["points_xyz"][lvid])
                Su_xyz = samp["Xu"][lvid]
                Sv_xyz = samp["Xv"][lvid]
                E = float(np.dot(Su_xyz, Su_xyz))
                Fm = float(np.dot(Su_xyz, Sv_xyz))
                G = float(np.dot(Sv_xyz, Sv_xyz))
                En, Fn, Gn = self.metric_EFG_normalized(E, Fm, G, *samp["u_raw_bounds"], *samp["v_raw_bounds"])
                det = (En * Gn) - (Fn * Fn)
                metric_valid = bool(np.isfinite(det) and det > float(self.det_min) and En > 0.0 and Gn > 0.0)
                mesh_rows.append({
                    "face_id": int(face_id),
                    "lvid": int(lvid),
                    "gvid": int(base_gvid + lvid),
                    "u": float(samp["uv"][lvid, 0]),
                    "v": float(samp["uv"][lvid, 1]),
                    "u_raw": u_raw,
                    "v_raw": v_raw,
                    "x": x,
                    "y": y,
                    "z": z,
                    "Su_x": float(Su_xyz[0]),
                    "Su_y": float(Su_xyz[1]),
                    "Su_z": float(Su_xyz[2]),
                    "Sv_x": float(Sv_xyz[0]),
                    "Sv_y": float(Sv_xyz[1]),
                    "Sv_z": float(Sv_xyz[2]),
                    "E": float(En),
                    "F": float(Fn),
                    "G": float(Gn),
                    "det": float(det),
                    "metric_valid": metric_valid,
                    "bbox_xmin": float(bbox["xmin"]),
                    "bbox_ymin": float(bbox["ymin"]),
                    "bbox_zmin": float(bbox["zmin"]),
                    "bbox_xmax": float(bbox["xmax"]),
                    "bbox_ymax": float(bbox["ymax"]),
                    "bbox_zmax": float(bbox["zmax"]),
                    "face_u_periodic": bool(samp["u_periodic"]),
                    "face_v_periodic": bool(samp["v_periodic"]),
                    "surface_u_periodic": bool(samp["u_periodic"]),
                    "surface_v_periodic": bool(samp["v_periodic"]),
                    "face_u_period": samp["u_period"],
                    "face_v_period": samp["v_period"],
                    "face_u_raw_min": float(samp["u_raw_bounds"][0]),
                    "face_u_raw_max": float(samp["u_raw_bounds"][1]),
                    "face_v_raw_min": float(samp["v_raw_bounds"][0]),
                    "face_v_raw_max": float(samp["v_raw_bounds"][1]),
                })

            for a, b, c in tri_faces:
                face_rows.append({
                    "face_id": int(face_id),
                    "i": int(base_gvid + a),
                    "j": int(base_gvid + b),
                    "k": int(base_gvid + c),
                })

        mesh_df = pd.DataFrame(mesh_rows)
        faces_df = pd.DataFrame(face_rows)
        return mesh_df, faces_df

    # =========================================================================
    # 6) Export mesh tables
    # =========================================================================
    def export_face_mesh_metric(self, shape_path: str, visualize: bool = False, visualize_face_id: int | None = None):
        """
        Returns:
            mesh_df: per-vertex table
            faces_df: global triangle table

        Notes:
            - `surface_*_periodic` describes the underlying OCC surface.
            - `face_*_periodic` describes whether the *trimmed face domain* should
            actually wrap for optimization/decoder use.
            - This distinction matters for trimmed periodic surfaces such as a
            half-cylinder patch: the underlying surface is periodic, but the face
            is not necessarily wrap-closed.
        """
        shape = self.load_shape(shape_path)
        self.mesh_shape_force(shape, deflection=self.deflection, angle=self.angle)

        mesh_rows = []
        face_rows = []
        face_lvid_to_gvid = {}

        for face_id, face in enumerate(self.iter_faces(shape)):
            (umin, umax, vmin, vmax), surf = self.face_uv_bounds_and_surface(face)
            surface_u_periodic, surface_v_periodic, u_period, v_period = self.face_uv_periodicity(face)

            tri = self.triangulate_face(face, n_u=self.n_u, n_v=self.n_v, tol=1e-7)
            if tri is None:
                print(f"[warn] face {face_id}: no triangulation, skipped")
                continue

            V_xyz, UV_raw, F_idx = tri

            valid_local_vertices = []
            u_raw_list = []
            v_raw_list = []

            for lvid, (xyz, uv) in enumerate(zip(V_xyz, UV_raw)):
                if uv is None or uv[0] is None or uv[1] is None:
                    continue

                u_raw = float(uv[0])
                v_raw = float(uv[1])
                x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])

                valid_local_vertices.append((lvid, x, y, z, u_raw, v_raw))
                u_raw_list.append(u_raw)
                v_raw_list.append(v_raw)

            if len(valid_local_vertices) == 0:
                print(f"[warn] face {face_id}: triangulation had no valid UV vertices, skipped")
                continue

            # Effective wrapping for the trimmed face domain.
            wrap_u = False
            wrap_v = False

            if surface_u_periodic and (u_period is not None):
                try:
                    wrap_u = bool(
                        self._should_wrap_u_true_seam(
                            face,
                            surf,
                            float(umin),
                            float(umax),
                            float(vmin),
                            float(vmax),
                            None,
                        )
                    )
                except Exception:
                    if len(u_raw_list) > 0:
                        uvals = np.asarray(u_raw_list, dtype=float)
                        umin_raw = float(np.min(uvals))
                        umax_raw = float(np.max(uvals))
                        span_u = umax_raw - umin_raw
                        wrap_u = abs(span_u - float(u_period)) <= max(1e-6, 0.05 * float(u_period))
                    else:
                        wrap_u = False

            if surface_v_periodic and (v_period is not None) and len(v_raw_list) > 0:
                vvals = np.asarray(v_raw_list, dtype=float)
                vmin_raw = float(np.min(vvals))
                vmax_raw = float(np.max(vvals))
                span_v = vmax_raw - vmin_raw
                wrap_v = abs(span_v - float(v_period)) <= max(1e-6, 0.05 * float(v_period))

            bbox = self.face_bounding_box(face)

            for lvid, x, y, z, u_raw, v_raw in valid_local_vertices:
                u_n, v_n = self.uv_raw_to_norm(u_raw, v_raw, umin, umax, vmin, vmax)

                gvid = len(mesh_rows)
                face_lvid_to_gvid[(face_id, lvid)] = gvid

                efg = self.metric_EFG_raw(surf, u_raw, v_raw, tol=self.metric_tol)
                if efg is None:
                    Su_xyz = (np.nan, np.nan, np.nan)
                    Sv_xyz = (np.nan, np.nan, np.nan)
                    En = Fn = Gn = np.nan
                    det = np.nan
                else:
                    Su_xyz, Sv_xyz, E, Fm, G = efg
                    En, Fn, Gn = self.metric_EFG_normalized(E, Fm, G, umin, umax, vmin, vmax)
                    det = (En * Gn) - (Fn * Fn)

                metric_valid = bool(
                    efg is not None
                    and np.isfinite(det)
                    and det > float(self.det_min)
                    and En > 0.0
                    and Gn > 0.0
                )

                mesh_rows.append({
                    "face_id": int(face_id),
                    "lvid": int(lvid),
                    "gvid": int(gvid),

                    "u": float(u_n),
                    "v": float(v_n),
                    "u_raw": float(u_raw),
                    "v_raw": float(v_raw),

                    "x": x,
                    "y": y,
                    "z": z,

                    "Su_x": float(Su_xyz[0]),
                    "Su_y": float(Su_xyz[1]),
                    "Su_z": float(Su_xyz[2]),
                    "Sv_x": float(Sv_xyz[0]),
                    "Sv_y": float(Sv_xyz[1]),
                    "Sv_z": float(Sv_xyz[2]),

                    "E": float(En),
                    "F": float(Fn),
                    "G": float(Gn),
                    "det": float(det),
                    "metric_valid": metric_valid,

                    "bbox_xmin": float(bbox["xmin"]),
                    "bbox_ymin": float(bbox["ymin"]),
                    "bbox_zmin": float(bbox["zmin"]),
                    "bbox_xmax": float(bbox["xmax"]),
                    "bbox_ymax": float(bbox["ymax"]),
                    "bbox_zmax": float(bbox["zmax"]),

                    "face_u_periodic": bool(wrap_u),
                    "face_v_periodic": bool(wrap_v),

                    "surface_u_periodic": bool(surface_u_periodic),
                    "surface_v_periodic": bool(surface_v_periodic),

                    "face_u_period": None if u_period is None else float(u_period),
                    "face_v_period": None if v_period is None else float(v_period),

                    "face_u_raw_min": float(umin),
                    "face_u_raw_max": float(umax),
                    "face_v_raw_min": float(vmin),
                    "face_v_raw_max": float(vmax),
                })

            for f in F_idx:
                a, b, c = int(f[0]), int(f[1]), int(f[2])
                ka, kb, kc = (face_id, a), (face_id, b), (face_id, c)
                if ka not in face_lvid_to_gvid or kb not in face_lvid_to_gvid or kc not in face_lvid_to_gvid:
                    continue

                face_rows.append({
                    "face_id": int(face_id),
                    "i": int(face_lvid_to_gvid[ka]),
                    "j": int(face_lvid_to_gvid[kb]),
                    "k": int(face_lvid_to_gvid[kc]),
                })

        mesh_df = pd.DataFrame(mesh_rows)
        faces_df = pd.DataFrame(face_rows)
        return mesh_df, faces_df    # =========================================================================
    @staticmethod
    def _empty_long(device):
        return torch.empty((0,), dtype=torch.long, device=device)
    @staticmethod
    def boundary_vertex_indices_from_faces_numpy(faces_ijk: torch.Tensor):
        f = faces_ijk.detach().cpu().numpy().astype(np.int64)
        edge_count = {}

        for a, b, c in f:
            e1 = (a, b) if a < b else (b, a)
            e2 = (b, c) if b < c else (c, b)
            e3 = (c, a) if c < a else (a, c)
            edge_count[e1] = edge_count.get(e1, 0) + 1
            edge_count[e2] = edge_count.get(e2, 0) + 1
            edge_count[e3] = edge_count.get(e3, 0) + 1

        boundary_verts = set()
        for (i, j), cnt in edge_count.items():
            if cnt == 1:
                boundary_verts.add(i)
                boundary_verts.add(j)

        boundary_idx = np.array(sorted(boundary_verts), dtype=np.int64)
        return torch.from_numpy(boundary_idx)

    @staticmethod
    def vertex_adjacency_from_faces(faces, num_verts):
        faces = faces.detach().cpu().numpy()
        adj = [set() for _ in range(num_verts)]
        for a, b, c in faces:
            adj[a].update([b, c])
            adj[b].update([a, c])
            adj[c].update([a, b])
        return adj

    @staticmethod
    def k_ring(start_idx: torch.Tensor, adj, k: int = 1) -> torch.Tensor:
        device = start_idx.device
        frontier = set(map(int, start_idx.detach().to("cpu").tolist()))
        visited = set(frontier)

        for _ in range(k):
            nxt = set()
            for v in frontier:
                nxt.update(adj[v])
            nxt -= visited
            visited |= nxt
            frontier = nxt
            if not frontier:
                break

        return torch.tensor(sorted(visited), dtype=torch.long, device=device)

    @staticmethod
    def precompute_face_areas(points_xyz: torch.Tensor, faces_ijk: torch.Tensor) -> torch.Tensor:
        i, j, k = faces_ijk[:, 0], faces_ijk[:, 1], faces_ijk[:, 2]
        pi, pj, pk = points_xyz[i], points_xyz[j], points_xyz[k]
        areas = 0.5 * torch.linalg.norm(torch.cross(pj - pi, pk - pi, dim=1), dim=1)
        return areas

    @staticmethod
    def material_amount_from_vertex_density(
        density_v: torch.Tensor,
        faces_ijk: torch.Tensor,
        face_areas: torch.Tensor,
    ) -> torch.Tensor:
        i, j, k = faces_ijk[:, 0], faces_ijk[:, 1], faces_ijk[:, 2]
        rho_f = (density_v[i] + density_v[j] + density_v[k]) / 3.0
        return (face_areas * rho_f).sum()

    @classmethod
    def compute_min_fraction(cls, points_xyz, faces_ijk, boundary_solid_idx):
        face_areas = cls.precompute_face_areas(points_xyz, faces_ijk)
        V0 = face_areas.sum()

        rho_min = torch.zeros(points_xyz.shape[0], device=points_xyz.device)
        rho_min[boundary_solid_idx] = 1.0

        V_min = cls.material_amount_from_vertex_density(rho_min, faces_ijk, face_areas)
        return float((V_min / (V0 + 1e-12)).item())

    @staticmethod
    def faces_df_to_pv_faces_autodetect(faces_df):
        num_cols = [c for c in faces_df.columns if np.issubdtype(faces_df[c].dtype, np.number)]
        if len(num_cols) < 3:
            raise ValueError("faces_df must have at least 3 numeric columns of vertex indices.")

        preferred_sets = [
            ["i", "j", "k", "l"],
            ["v0", "v1", "v2", "v3"],
            ["a", "b", "c", "d"],
            ["n0", "n1", "n2", "n3"],
        ]

        cols = None
        for pref in preferred_sets:
            have = [c for c in pref if c in faces_df.columns]
            if len(have) >= 3:
                cols = have[:4]
                break

        if cols is None:
            cols = num_cols[:4]

        arr = faces_df[cols].to_numpy()
        is_quad_candidate = arr.shape[1] >= 4
        quad_mask = ~np.isnan(arr[:, 3]) if is_quad_candidate else None

        faces_list = []

        def _emit(face_idx):
            faces_list.extend(face_idx.tolist())

        if quad_mask is None:
            tri = arr[:, :3].astype(np.int64)
            for f in tri:
                _emit(np.array([3, f[0], f[1], f[2]], dtype=np.int64))
        else:
            tri_rows = arr[~quad_mask, :3]
            quad_rows = arr[quad_mask, :4]

            for f in tri_rows:
                f = f.astype(np.int64)
                _emit(np.array([3, f[0], f[1], f[2]], dtype=np.int64))

            for f in quad_rows:
                f = f.astype(np.int64)
                _emit(np.array([4, f[0], f[1], f[2], f[3]], dtype=np.int64))

        faces = np.asarray(faces_list, dtype=np.int64)

        idx_only = faces[faces != 3]
        idx_only = idx_only[idx_only != 4]
        if idx_only.size and idx_only.min() == 1 and not np.any(idx_only == 0):
            out = faces.copy()
            k = 0
            while k < out.size:
                n = out[k]
                out[k + 1:k + 1 + n] -= 1
                k += 1 + n
            faces = out

        return faces

    def _build_single_face_tensor_dict(self, mesh_face_df, faces_face_df, input_ring: int):
        mesh_face_df = mesh_face_df.sort_values("gvid").reset_index(drop=True)
        device = self.device

        global_vertex_idx = torch.tensor(
            mesh_face_df["gvid"].to_numpy(),
            dtype=torch.long,
            device=device,
        )

        uv = torch.tensor(
            mesh_face_df[["u", "v"]].to_numpy(),
            dtype=torch.float32,
            device=device,
        )
        points_xyz = torch.tensor(
            mesh_face_df[["x", "y", "z"]].to_numpy(),
            dtype=torch.float32,
            device=device,
        )
        Xu = torch.tensor(
            mesh_face_df[["Su_x", "Su_y", "Su_z"]].to_numpy(),
            dtype=torch.float32,
            device=device,
        )
        Xv = torch.tensor(
            mesh_face_df[["Sv_x", "Sv_y", "Sv_z"]].to_numpy(),
            dtype=torch.float32,
            device=device,
        )

        gvid_to_local = {int(g): i for i, g in enumerate(mesh_face_df["gvid"].tolist())}
        if len(faces_face_df) > 0:
            faces_local_np = faces_face_df[["i", "j", "k"]].replace(gvid_to_local).to_numpy(dtype=np.int64)
            faces_ijk = torch.tensor(faces_local_np, dtype=torch.long, device=device)
            global_face_idx = torch.tensor(faces_face_df.index.to_numpy(), dtype=torch.long, device=device)
        else:
            faces_ijk = torch.empty((0, 3), dtype=torch.long, device=device)
            global_face_idx = self._empty_long(device)

        pv_faces = self.faces_ijk_to_pv_faces(faces_ijk) if faces_ijk.numel() else np.empty((0,), dtype=np.int64)
        num_verts = int(uv.shape[0])

        if faces_ijk.numel():
            boundary_idx = self.boundary_vertex_indices_from_faces_numpy(faces_ijk).to(device=device)
            adj = self.vertex_adjacency_from_faces(faces_ijk, num_verts)
            boundary_idx_ring1 = self.k_ring(boundary_idx, adj, k=input_ring)
            boundary_idx_ring2 = self.k_ring(boundary_idx, adj, k=8)
            face_areas = self.precompute_face_areas(points_xyz, faces_ijk)
            min_vol_frac = self.compute_min_fraction(points_xyz, faces_ijk, boundary_idx_ring1)
        else:
            boundary_idx = self._empty_long(device)
            boundary_idx_ring1 = self._empty_long(device)
            boundary_idx_ring2 = self._empty_long(device)
            face_areas = torch.empty((0,), dtype=torch.float32, device=device)
            min_vol_frac = 0.0

        row0 = mesh_face_df.iloc[0]
        bbox = {
            "xmin": float(row0["bbox_xmin"]),
            "xmax": float(row0["bbox_xmax"]),
            "ymin": float(row0["bbox_ymin"]),
            "ymax": float(row0["bbox_ymax"]),
            "zmin": float(row0["bbox_zmin"]),
            "zmax": float(row0["bbox_zmax"]),
        }

        fid = int(row0["face_id"])
        return {
            "face_id": fid,
            "uv": uv,
            "points_xyz": points_xyz,
            "Xu": Xu,
            "Xv": Xv,
            "faces_ijk": faces_ijk,
            "pv_faces": pv_faces,
            "face_areas": face_areas,
            "boundary_idx": boundary_idx,
            "boundary_idx_ring1": boundary_idx_ring1,
            "boundary_idx_ring2": boundary_idx_ring2,
            "min_vol_frac": float(min_vol_frac),
            "BBX": bbox,
            "u_periodic": bool(row0["face_u_periodic"]),
            "v_periodic": bool(row0["face_v_periodic"]),
            "u_period": None if pd.isna(row0["face_u_period"]) else float(row0["face_u_period"]),
            "v_period": None if pd.isna(row0["face_v_period"]) else float(row0["face_v_period"]),
            "u_raw_bounds": (float(row0["face_u_raw_min"]), float(row0["face_u_raw_max"])),
            "v_raw_bounds": (float(row0["face_v_raw_min"]), float(row0["face_v_raw_max"])),
            "global_vertex_idx": global_vertex_idx,
            "global_face_idx": global_face_idx,
            "num_vertices": num_verts,
            "num_faces": int(faces_ijk.shape[0]),
        }

    def generate_input_tensors_from_dataframes(self, mesh_df, faces_df, input_ring: int):
        mesh_df_ord = mesh_df.sort_values("gvid").reset_index(drop=True)

        uv = torch.tensor(
            mesh_df_ord[["u", "v"]].to_numpy(),
            dtype=torch.float32,
            device=self.device,
        )
        points_xyz = torch.tensor(
            mesh_df_ord[["x", "y", "z"]].to_numpy(),
            dtype=torch.float32,
            device=self.device,
        )
        Xu = torch.tensor(
            mesh_df_ord[["Su_x", "Su_y", "Su_z"]].to_numpy(),
            dtype=torch.float32,
            device=self.device,
        )
        Xv = torch.tensor(
            mesh_df_ord[["Sv_x", "Sv_y", "Sv_z"]].to_numpy(),
            dtype=torch.float32,
            device=self.device,
        )
        faces_ijk = torch.tensor(
            faces_df[["i", "j", "k"]].to_numpy(),
            dtype=torch.long,
            device=self.device,
        )

        num_verts = uv.shape[0]
        pv_faces = self.faces_df_to_pv_faces_autodetect(faces_df)

        if faces_ijk.numel():
            boundary_idx = self.boundary_vertex_indices_from_faces_numpy(faces_ijk).to(device=uv.device)
            adj = self.vertex_adjacency_from_faces(faces_ijk, num_verts)
            boundary_idx_ring1 = self.k_ring(boundary_idx, adj, k=input_ring)
            boundary_idx_ring2 = self.k_ring(boundary_idx, adj, k=8)
            face_areas = self.precompute_face_areas(points_xyz, faces_ijk)
            min_vol_frac = self.compute_min_fraction(points_xyz, faces_ijk, boundary_idx_ring1)
        else:
            boundary_idx = self._empty_long(self.device)
            boundary_idx_ring1 = self._empty_long(self.device)
            boundary_idx_ring2 = self._empty_long(self.device)
            face_areas = torch.empty((0,), dtype=torch.float32, device=self.device)
            min_vol_frac = 0.0

        face_id = torch.tensor(
            mesh_df_ord["face_id"].to_numpy(),
            dtype=torch.long,
            device=self.device,
        )

        BBX = {}
        face_tensors = []
        face_tensors_by_id = {}
        face_periodicity = {}

        for fid, grp in mesh_df_ord.groupby("face_id", sort=True):
            fid = int(fid)

            xmin = float(grp["x"].min())
            xmax = float(grp["x"].max())
            ymin = float(grp["y"].min())
            ymax = float(grp["y"].max())
            zmin = float(grp["z"].min())
            zmax = float(grp["z"].max())

            BBX[fid] = {
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax,
                "zmin": zmin,
                "zmax": zmax,
            }

            faces_face_df = faces_df[faces_df["face_id"] == fid].copy().reset_index(drop=False)
            face_dict = self._build_single_face_tensor_dict(grp.copy(), faces_face_df, input_ring=input_ring)

            face_tensors.append(face_dict)
            face_tensors_by_id[fid] = face_dict
            face_periodicity[fid] = {
                "u_periodic": face_dict["u_periodic"],
                "v_periodic": face_dict["v_periodic"],
                "u_period": face_dict["u_period"],
                "v_period": face_dict["v_period"],
            }

        print("MinVolFrac:", min_vol_frac)

        return {
            "uv": uv,
            "points_xyz": points_xyz,
            "face_areas": face_areas,
            "Xu": Xu,
            "Xv": Xv,
            "faces_ijk": faces_ijk,
            "pv_faces": pv_faces,
            "face_id": face_id,
            "boundary_idx": boundary_idx,
            "boundary_idx_ring1": boundary_idx_ring1,
            "boundary_idx_ring2": boundary_idx_ring2,
            "min_vol_frac": min_vol_frac,
            "BBX": BBX,
            "face_tensors": face_tensors,
            "face_tensors_by_id": face_tensors_by_id,
            "face_periodicity": face_periodicity,
            "num_faces": len(face_tensors),
        }
    def generate_from_file_points_triangulated(
        self,
        shape_path: str,
        input_ring: int,
        M_per_face: int,
        pool_size_factor: int = 10,
        fps_pool_factor: int = 4,
        use_fps: bool = True,
        triangulation_max_edge_rel: float = 0.35,
    ):
        mesh_df, faces_df = self.export_face_mesh_metric_points_triangulated(
            shape_path=shape_path,
            M_per_face=M_per_face,
            pool_size_factor=pool_size_factor,
            fps_pool_factor=fps_pool_factor,
            use_fps=use_fps,
            triangulation_max_edge_rel=triangulation_max_edge_rel,
        )
        tensors = self.generate_input_tensors_from_dataframes(mesh_df, faces_df, input_ring=input_ring)
        return mesh_df, faces_df, tensors

    def generate_from_file(
        self,
        shape_path: str,
        input_ring: int,
        visualize: bool = False,
        visualize_face_id: int | None = None,
        mode: str = "mesh",
        M_per_face: int | None = None,
        pool_size_factor: int = 10,
        fps_pool_factor: int = 4,
        use_fps: bool = True,
        triangulation_max_edge_rel: float = 0.35,
    ):
        """
        Full pipeline:
          CAD file -> mesh_df/faces_df -> tensors

        mode:
          - "mesh": original OCC/manual triangulation path
          - "points_triangulated": area-weighted surface sampling + UV triangulation
        """
        if mode == "mesh":
            mesh_df, faces_df = self.export_face_mesh_metric(
                shape_path=shape_path,
                visualize=visualize,
                visualize_face_id=visualize_face_id,
            )
        elif mode == "Sampled_points":
            if M_per_face is None:
                raise ValueError("M_per_face must be provided when mode='points_triangulated'.")
            mesh_df, faces_df = self.export_face_mesh_metric_points_triangulated(
                shape_path=shape_path,
                M_per_face=M_per_face,
                pool_size_factor=pool_size_factor,
                fps_pool_factor=fps_pool_factor,
                use_fps=use_fps,
                triangulation_max_edge_rel=triangulation_max_edge_rel,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        tensors = self.generate_input_tensors_from_dataframes(mesh_df, faces_df, input_ring=input_ring)
        return mesh_df, faces_df, tensors
    @staticmethod
    def faces_ijk_to_pv_faces(faces_ijk: torch.Tensor) -> np.ndarray:
        """
        Convert triangle faces tensor (F,3) to PyVista flat faces array (4F,).
        Format: [3, i, j, k, 3, i, j, k, ...]
        """
        f = faces_ijk.detach().cpu().numpy().astype(np.int64)  # (F,3)
        F = f.shape[0]
        pv_faces = np.empty((F, 4), dtype=np.int64)
        pv_faces[:, 0] = 3
        pv_faces[:, 1:] = f
        return pv_faces.reshape(-1)

    @staticmethod
    @torch.no_grad()
    def seeds_uv_to_xyz_nearest(
        seeds_uv: torch.Tensor,
        uv: torch.Tensor,
        points_xyz: torch.Tensor
    ):
        """
        Map UV seeds to nearest XYZ points.

        seeds_uv:   (S,2) in [0,1]
        uv:         (N,2)
        points_xyz: (N,3)

        returns:
            (S,3) numpy array for plotting
        """
        d2 = torch.cdist(seeds_uv, uv).pow(2)   # (S,N)
        idx = d2.argmin(dim=1)                  # (S,)
        return points_xyz[idx].detach().cpu().numpy()

    @staticmethod
    def fps_3d(
        points_xyz: torch.Tensor,
        S: int,
        exclude_idx: torch.Tensor | None = None,
        start_idx: int | None = None,
    ) -> torch.Tensor:
        """
        Farthest Point Sampling in 3D.

        points_xyz:  (N,3) torch tensor
        S:           number of samples
        exclude_idx: (E,) long tensor of forbidden vertex indices
        start_idx:   optional start vertex index (global)

        returns:
            idx_global: (S,) long tensor of sampled vertex indices
        """
        device = points_xyz.device
        N = points_xyz.shape[0]

        valid = torch.ones(N, dtype=torch.bool, device=device)
        if exclude_idx is not None and exclude_idx.numel() > 0:
            valid[exclude_idx] = False

        valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
        if valid_idx.numel() < S:
            raise ValueError(f"Not enough valid points ({valid_idx.numel()}) for S={S}")

        P = points_xyz[valid_idx]  # (M,3)
        M = P.shape[0]

        idx_local = torch.empty(S, dtype=torch.long, device=device)

        if start_idx is None:
            idx_local[0] = torch.randint(0, M, (1,), device=device)
        else:
            pos = (valid_idx == start_idx).nonzero(as_tuple=False)
            if pos.numel() == 0:
                raise ValueError("start_idx is excluded or not valid.")
            idx_local[0] = pos.item()

        d2 = torch.cdist(P[idx_local[0]].unsqueeze(0), P).squeeze(0).pow(2)

        for i in range(1, S):
            idx_local[i] = torch.argmax(d2)
            d2 = torch.minimum(
                d2,
                torch.cdist(P[idx_local[i]].unsqueeze(0), P).squeeze(0).pow(2)
            )

        idx_global = valid_idx[idx_local]
        return idx_global

    @staticmethod
    def fps_uv_avoid_boundary_band(
        uv: torch.Tensor,
        S: int,
        boundary_idx: torch.Tensor,
        margin_uv: float = 0.05,
        start_idx: int | None = None,
    ):
        """
        FPS in UV space while excluding a boundary band.

        Excludes any uv point within 'margin_uv' of any boundary uv point.

        uv:           (N,2)
        S:            number of samples
        boundary_idx: indices of boundary points
        margin_uv:    minimum UV distance from boundary
        start_idx:    optional global start vertex index

        returns:
            selected_uv:  (S,2)
            idx_sel:      (S,)
        """
        device = uv.device
        boundary_uv = uv[boundary_idx]  # (B,2)

        dmin = torch.cdist(uv, boundary_uv).min(dim=1).values  # (N,)
        valid = dmin >= margin_uv
        valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)

        if valid_idx.numel() < S:
            raise ValueError(
                f"Not enough interior points ({valid_idx.numel()}) for S={S} with margin_uv={margin_uv}."
            )

        uv_valid = uv[valid_idx]
        M = uv_valid.shape[0]

        idx_local = torch.empty(S, dtype=torch.long, device=device)

        if start_idx is None:
            idx_local[0] = torch.randint(0, M, (1,), device=device)
        else:
            pos = (valid_idx == start_idx).nonzero(as_tuple=False)
            if pos.numel() == 0:
                raise ValueError("start_idx is outside valid band.")
            idx_local[0] = pos.item()

        d2 = torch.cdist(uv_valid[idx_local[0]].unsqueeze(0), uv_valid).squeeze(0).pow(2)

        for i in range(1, S):
            idx_local[i] = torch.argmax(d2)
            d2 = torch.minimum(
                d2,
                torch.cdist(uv_valid[idx_local[i]].unsqueeze(0), uv_valid).squeeze(0).pow(2)
            )

        idx_sel = valid_idx[idx_local]
        return uv[idx_sel], idx_sel

    @staticmethod
    def tau_schedule(step, max_steps, tau0=0.5, tau1=0.005):
        t = step / max(1, max_steps - 1)
        return tau0 * (tau1 / tau0) ** t

    @staticmethod
    def beta_schedule(step, max_steps, beta0=0.08, beta1=0.015):
        t = step / max(1, max_steps - 1)
        if t < 0.7:
            return beta0
        tb = (t - 0.7) / 0.3
        return beta0 * (beta1 / beta0) ** tb
    @staticmethod
    def vertex_area_lumped(N, faces_ijk, face_areas):
        """Lumped vertex area weights A_v (N,). Each triangle area distributed equally to its 3 vertices."""
        A_v = torch.zeros((N,), device=faces_ijk.device, dtype=face_areas.dtype)
        a = face_areas / 3.0
        tri = faces_ijk
        A_v.index_add_(0, tri[:, 0], a)
        A_v.index_add_(0, tri[:, 1], a)
        A_v.index_add_(0, tri[:, 2], a)
        return A_v
    @staticmethod
    def face_bounding_box(face, use_triangulation: bool = True):
        """
        Compute axis-aligned 3D bounding box of a face.

        Returns:
            {
                "xmin": ...,
                "xmax": ...,
                "ymin": ...,
                "ymax": ...,
                "zmin": ...,
                "zmax": ...,
            }
        """
        box = Bnd_Box()
        brepbndlib.Add(face, box, use_triangulation)

        if box.IsVoid():
            raise RuntimeError("Bounding box is void.")

        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
        return {
            "xmin": float(xmin),
            "xmax": float(xmax),
            "ymin": float(ymin),
            "ymax": float(ymax),
            "zmin": float(zmin),
            "zmax": float(zmax),
        }    