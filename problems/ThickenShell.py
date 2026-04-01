import os.path
import math
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
import pyvista as pv
try:
    pv.set_jupyter_backend("trame")
except Exception:
    pass
import torch

from .problemBase import problemBase


class ThickenShell(problemBase):
    problemName = 'ThickenShell'

    def __init__(self, thickness,BC_dir,Load_magnitude, voxel_size, extra_layers=1, tensors=None,tangential_tol=None):
        super().__init__()
        self.name = self.problemName

        self.brep_bbox = None
        self.thickness = float(thickness)
        self.face_bboxes = None
        self.samples_by_face = None
        self.voxel_size = float(voxel_size)
        self.extra_layers = int(extra_layers)
        self.tangential_tol = None if tangential_tol is None else float(tangential_tol)
        self.BC_dir = BC_dir
        self.Load_magnitude = Load_magnitude

        self.grid_geom = None
        self.elem_centers = None
        self.node_coords = None

        self.elem_sample_idx = None
        self.elem_fiber = None
        self.elem_phi = None
        self.elem_theta = None

        self.uv = None
        self.points_xyz = None
        self.face_areas = None
        self.Xu = None
        self.Xv = None
        self.faces_ijk = None
        self.pv_faces = None
        self.face_id = None
        self.boundary_idx_ring1 = None
        self.min_vol_frac = None
        self.sample_normals = None

        self.elem_occupancy = None
        self.elem_density = None

        if tensors is None:
            raise ValueError("tensors must be provided at this stage")

        # Get CAD info about the shell geometry and samples from the tensors. Sample: points_xyz, Xu, Xv, face_areas, etc. Also parse the bounding box of the BREP geometry.
        self.set_cad_samples(tensors)
   

        # Build a full structured voxel grid that covers the padded bounding box of the shell.
        # This creates a rectangular grid (mesh), initializes empty boundary conditions,
        # and defines material properties. At this stage NO voxels are trimmed or filtered
        # based on the shell geometry — the grid is still a complete box. The actual shell
        # shape is imposed later by voxelize_shell_from_samples(), which marks which voxels
        # belong to the shell thickness via elem_occupancy.
        self.mesh, self.boundaryCondition, self.materialProperty = self.shellSettings()



        # Voxelize the shell midsurface samples into the structured voxel grid.
        # For each voxel center, we find the nearest sampled point on the CAD surface
        # The distance from the voxel center to the surface sample is decomposed into:
        #   - normal distance (dn): distance along the surface normal
        #   - tangential distance (dt): distance in the surface tangent plane
        # A voxel is marked as part of the shell if:
        #   dn <= thickness/2                (within shell thickness)
        #   dt <= tangential_tol             (close enough along the surface)
        #   de <= max_euclid                 (overall distance safety bound)
        # Outputs:
        #   occ        : voxel occupancy grid (1 = shell voxel, 0 = empty)
        #   sample_idx : index of the nearest surface sample for each voxel
        self.elem_occupancy,self.elem_sample_idx = self.voxelize_shell_from_samples(self.thickness,tangential_tol=self.tangential_tol)


        # Create a simple binary density field based on voxel occupancy.
        # Occupied voxels (shell) get density ≈ 1, empty voxels get a small void density rho_min.
        # This is only an initialization useful for debugging / visualization.
        # The actual density and fiber fields used in the FEM solve will later come
        # from the neural decoder and can be assigned via assign_decoder_fields().
        rho_min = 1e-3
        self.elem_density = rho_min + (1.0 - rho_min) * self.elem_occupancy.reshape(-1).astype(np.float32)


        # Select nodes on the bottom and top of the voxelized shell to apply boundary conditions.
        # We define thin slabs near the minimum and maximum z of the padded bounding box.
        # Nodes within these slabs will be used for supports and loads.
        tol = 2 * self.voxel_size

        # Nodes belonging to voxels that are part of the shell (avoid selecting nodes in empty space)
        shell_nodes = self.occupied_node_ids()

        # Use the actual occupied shell extent rather than the full padded grid box.
        # This works better for multi-face inputs and avoids selecting nodes from empty padding.
        bbox = self.occupied_axis_bounds()
        # Recompute the padded bounding box used to build the voxel grid
        bbox = self.padded_bbox_from_midsurface(self.brep_bbox, self.thickness, self.voxel_size, self.extra_layers)

        force_box_nodes= None
        force_nodes = None

        

        if(self.BC_dir == "x"):
                    # Nodes in a small region near the bottom of the grid → candidate fixed support nodes
            fixed_box_nodes = self.select_nodes_in_box(
                xmin=bbox['xmin'] ,
                xmax=bbox['xmin'] + tol
            )

            # Nodes in a small region near the top of the grid → candidate load nodes
            force_box_nodes = self.select_nodes_in_box(
                xmin=bbox['xmax'] - tol,
                xmax=bbox['xmax'] ,
            )
            
        elif (self.BC_dir == "y"):

            fixed_box_nodes = self.select_nodes_in_box(
                ymin=bbox['ymin'] ,
                ymax=bbox['ymin'] + tol
            )

            # Nodes in a small region near the top of the grid → candidate load nodes
            force_box_nodes = self.select_nodes_in_box(
                ymin=bbox['ymax'] - tol,
                ymax=bbox['ymax'] ,
            )

        elif (self.BC_dir == "z"):

            fixed_box_nodes = self.select_nodes_in_box(
                zmin=bbox['zmin'] ,
                zmax=bbox['zmin'] + tol
            )

            # Nodes in a small region near the top of the grid → candidate load nodes
            force_box_nodes = self.select_nodes_in_box(
                zmin=bbox['zmax'] - tol,
                zmax=bbox['zmax'] ,
            )
        else:
           raise ValueError(f"Unsupported BC_dir: {self.BC_dir}")

        # Restrict the above selections to nodes that actually belong to shell voxels
        # (removes nodes from empty parts of the padded grid)
        fixed_nodes = self.intersect_node_sets(shell_nodes, fixed_box_nodes)
        force_nodes = self.intersect_node_sets(shell_nodes, force_box_nodes)

        # Build the FEM boundary condition object:
        # - fix all DOFs of nodes near the bottom
        # - apply a distributed force in the +z direction to nodes near the top
        self.set_boundary_conditions_from_regions(
            fixed_nodes=fixed_nodes,
            force_nodes=force_nodes,
            force_direction=self.BC_dir,
            force_value=self.Load_magnitude,
        )

    def shellSettings(self):
        mesh, grid_geom, elem_centers, node_coords = self.build_voxel_grid_for_shell(
            self.brep_bbox,
            self.thickness,
            self.voxel_size,
            self.extra_layers
        )

        self.grid_geom = grid_geom
        self.elem_centers = elem_centers
        self.node_coords = node_coords

        # matProp = {
        #     'E': 1.0,
        #     'nu': 0.3,
        #     'Ef': 1.0,
        #     'Et': 1.0,
        #     'nuf': 0.3,
        #     'nut': 0.3,
        #     'penal': 3
        # }
        matProp = {
            'E': 1.0,
            'nu': 0.3,
            'Ef': 10.0,
            'Et': 1.0,
            'nuf': 0.25,
            'nut': 0.3,
            'penal': 3
        }

        ndof = 3 * (mesh['nelx'] + 1) * (mesh['nely'] + 1) * (mesh['nelz'] + 1)
        force = np.zeros((ndof, 1), dtype=float)
        fixed = np.array([], dtype=np.int64)

        bc = {
            'exampleName': self.name,
            'physics': 'Structural',
            'force': force,
            'fixed': fixed,
            'numDOFPerNode': 3
        }

        return mesh, bc, matProp

    def to_numpy(self, x):
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)

    def _normalize_single_bbox(self, bbox):
        required = ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax')
        if not isinstance(bbox, dict) or not all(k in bbox for k in required):
            raise ValueError(f"Unsupported single-face BBX format: {bbox}")
        return {
            'xmin': float(bbox['xmin']),
            'xmax': float(bbox['xmax']),
            'ymin': float(bbox['ymin']),
            'ymax': float(bbox['ymax']),
            'zmin': float(bbox['zmin']),
            'zmax': float(bbox['zmax']),
        }
    def parse_bbox(self, bbox_raw):
        """
        Accept either:
        - {'xmin':..., 'xmax':..., 'ymin':..., 'ymax':..., 'zmin':..., 'zmax':...}
        - {0: {...}, 1: {...}, ...}

        Returns
        -------
        union_bbox : dict
            Global union bounding box across all faces.
        face_bboxes : dict[int, dict]
            Per-face bounding boxes. For a single-face input, face id 0 is used.
        """
        if not isinstance(bbox_raw, dict):
            raise ValueError(f"Unsupported BBX format: {bbox_raw}")

        required = ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax')
        if all(k in bbox_raw for k in required):
            b = self._normalize_single_bbox(bbox_raw)
            return b, {0: b.copy()}

        face_bboxes = {}
        for fid, bbox in bbox_raw.items():
            if not isinstance(bbox, dict):
                raise ValueError(f"Unsupported BBX entry for face {fid}: {bbox}")
            face_bboxes[int(fid)] = self._normalize_single_bbox(bbox)

        if len(face_bboxes) == 0:
            raise ValueError(f"Unsupported empty BBX format: {bbox_raw}")

        union_bbox = {
            'xmin': min(b['xmin'] for b in face_bboxes.values()),
            'xmax': max(b['xmax'] for b in face_bboxes.values()),
            'ymin': min(b['ymin'] for b in face_bboxes.values()),
            'ymax': max(b['ymax'] for b in face_bboxes.values()),
            'zmin': min(b['zmin'] for b in face_bboxes.values()),
            'zmax': max(b['zmax'] for b in face_bboxes.values()),
        }
        return union_bbox, face_bboxes

    def set_cad_samples(self, tensors):
        self.uv = self.to_numpy(tensors["uv"])
        self.points_xyz = self.to_numpy(tensors["points_xyz"]).reshape(-1, 3)
        self.face_areas = self.to_numpy(tensors["face_areas"])
        self.Xu = self.to_numpy(tensors["Xu"]).reshape(-1, 3)
        self.Xv = self.to_numpy(tensors["Xv"]).reshape(-1, 3)
        self.faces_ijk = self.to_numpy(tensors["faces_ijk"])
        self.pv_faces = self.to_numpy(tensors["pv_faces"])
        self.face_id = self.to_numpy(tensors["face_id"]).reshape(-1).astype(np.int64)
        self.boundary_idx_ring1 = self.to_numpy(tensors["boundary_idx_ring1"])
        self.min_vol_frac = self.to_numpy(tensors["min_vol_frac"])

        self.sample_normals = self.compute_sample_normals(self.Xu, self.Xv)

        bbox_raw = tensors["BBX"]
        self.brep_bbox, self.face_bboxes = self.parse_bbox(bbox_raw)

        self.samples_by_face = {}
        if self.face_id.shape[0] != self.points_xyz.shape[0]:
            raise ValueError("face_id must have same length as points_xyz")
        for fid in np.unique(self.face_id):
            self.samples_by_face[int(fid)] = np.flatnonzero(self.face_id == fid).astype(np.int64)

    def occupied_node_ids(self):
        occ = self.elem_occupancy.astype(bool)   # shape (nelz, nelx, nely)
        nelz, nelx, nely = occ.shape

        node_mask = np.zeros((nelz + 1, nelx + 1, nely + 1), dtype=bool)

        for k in range(nelz):
            for i in range(nelx):
                for j in range(nely):
                    if occ[k, i, j]:
                        node_mask[k:k+2, i:i+2, j:j+2] = True

        node_ids = np.arange(node_mask.size, dtype=np.int64).reshape(node_mask.shape)
        return node_ids[node_mask]    
    def intersect_node_sets(self, a, b):
        return np.intersect1d(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64))

    def compute_sample_normals(self, Xu, Xv, eps=1e-12):
        normals = np.cross(Xu, Xv)
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.clip(norm, eps, None)
        return normals

    def voxel_center_is_near_bbox(self, center, bbox, margin):
        return (
            (bbox['xmin'] - margin <= center[0] <= bbox['xmax'] + margin) and
            (bbox['ymin'] - margin <= center[1] <= bbox['ymax'] + margin) and
            (bbox['zmin'] - margin <= center[2] <= bbox['zmax'] + margin)
        )

    def candidate_face_ids_for_center(self, center, margin):
        if not self.face_bboxes:
            return []
        return [
            fid for fid, bbox in self.face_bboxes.items()
            if self.voxel_center_is_near_bbox(center, bbox, margin)
        ]
    def voxelize_shell_from_samples(self, thickness, tangential_tol=None):
        centers = self.elem_centers.reshape(-1, 3)
        points = self.points_xyz
        normals = self.sample_normals

        if tangential_tol is None:
            tangential_tol = 0.35 * self.voxel_size

        half_t = 0.5 * thickness
        max_euclid = np.sqrt(half_t * half_t + tangential_tol * tangential_tol)
        bbox_margin = half_t + tangential_tol + self.voxel_size

        occ = np.zeros((centers.shape[0],), dtype=np.uint8)
        sample_idx = -np.ones((centers.shape[0],), dtype=np.int64)

        for i, x in enumerate(centers):
            candidate_face_ids = self.candidate_face_ids_for_center(x, bbox_margin)

            if candidate_face_ids:
                candidate_idx = np.concatenate([
                    self.samples_by_face[fid] for fid in candidate_face_ids if fid in self.samples_by_face
                ])
            else:
                candidate_idx = np.arange(points.shape[0], dtype=np.int64)

            if candidate_idx.size == 0:
                continue

            candidate_points = points[candidate_idx]
            diff = candidate_points - x[None, :]
            dist2 = np.einsum('ij,ij->i', diff, diff)
            local_j = np.argmin(dist2)
            j = candidate_idx[local_j]

            p = points[j]
            n = normals[j]

            r = x - p
            de = np.linalg.norm(r)
            dn = abs(np.dot(r, n))
            rt = r - np.dot(r, n) * n
            dt = np.linalg.norm(rt)

            if dn <= half_t and dt <= tangential_tol and de <= max_euclid:
                occ[i] = 1
                sample_idx[i] = j

        occ = occ.reshape(self.elem_centers.shape[:3])
        sample_idx = sample_idx.reshape(self.elem_centers.shape[:3])

        return occ, sample_idx

    def padded_bbox_from_midsurface(self, bbox, thickness, voxel_size, extra_layers=1):
        pad = thickness / 2.0 + extra_layers * voxel_size

        return {
            'xmin': bbox['xmin'] - pad,
            'xmax': bbox['xmax'] + pad,
            'ymin': bbox['ymin'] - pad,
            'ymax': bbox['ymax'] + pad,
            'zmin': bbox['zmin'] - pad,
            'zmax': bbox['zmax'] + pad,
        }

    def structured_grid_from_bbox(self, bbox, voxel_size):
        hx = hy = hz = float(voxel_size)

        lx = bbox['xmax'] - bbox['xmin']
        ly = bbox['ymax'] - bbox['ymin']
        lz = bbox['zmax'] - bbox['zmin']

        nelx = int(math.ceil(lx / hx))
        nely = int(math.ceil(ly / hy))
        nelz = int(math.ceil(lz / hz))

        mesh = {
            'nelx': nelx,
            'nely': nely,
            'nelz': nelz,
            'elemSize': np.array([hx, hy, hz], dtype=float),
            'type': 'grid'
        }

        grid_geom = {
            'xmin': bbox['xmin'],
            'ymin': bbox['ymin'],
            'zmin': bbox['zmin'],
            'hx': hx,
            'hy': hy,
            'hz': hz
        }

        return mesh, grid_geom

    def element_centers(self, mesh, grid_geom):
        nelx, nely, nelz = mesh['nelx'], mesh['nely'], mesh['nelz']
        hx, hy, hz = grid_geom['hx'], grid_geom['hy'], grid_geom['hz']
        xmin, ymin, zmin = grid_geom['xmin'], grid_geom['ymin'], grid_geom['zmin']

        xs = xmin + (np.arange(nelx) + 0.5) * hx
        ys = ymin + (np.arange(nely) + 0.5) * hy
        zs = zmin + (np.arange(nelz) + 0.5) * hz

        Z, X, Y = np.meshgrid(zs, xs, ys, indexing='ij')
        centers = np.stack([X, Y, Z], axis=-1)
        return centers

    def node_coordinates(self, mesh, grid_geom):
        nelx, nely, nelz = mesh['nelx'], mesh['nely'], mesh['nelz']
        hx, hy, hz = grid_geom['hx'], grid_geom['hy'], grid_geom['hz']
        xmin, ymin, zmin = grid_geom['xmin'], grid_geom['ymin'], grid_geom['zmin']

        xs = xmin + np.arange(nelx + 1) * hx
        ys = ymin + np.arange(nely + 1) * hy
        zs = zmin + np.arange(nelz + 1) * hz

        Z, X, Y = np.meshgrid(zs, xs, ys, indexing='ij')
        coords = np.stack([X, Y, Z], axis=-1)
        return coords

    def build_voxel_grid_for_shell(self, brep_bbox, thickness, voxel_size, extra_layers=1):
        padded = self.padded_bbox_from_midsurface(
            brep_bbox,
            thickness=thickness,
            voxel_size=voxel_size,
            extra_layers=extra_layers
        )

        mesh, grid_geom = self.structured_grid_from_bbox(padded, voxel_size)
        elem_centers = self.element_centers(mesh, grid_geom)
        node_coords = self.node_coordinates(mesh, grid_geom)

        return mesh, grid_geom, elem_centers, node_coords
    
    def assign_surface_fields_to_voxels(self, rho_surface, fiber_surface, rho_void=1e-3):
        rho_surface = self.to_numpy(rho_surface).reshape(-1)
        fiber_surface = self.to_numpy(fiber_surface).reshape(-1, 3)

        if rho_surface.shape[0] != self.points_xyz.shape[0]:
            raise ValueError("rho_surface must have same length as points_xyz")

        if fiber_surface.shape[0] != self.points_xyz.shape[0]:
            raise ValueError("fiber_surface must have same length as points_xyz")

        sample_idx = self.elem_sample_idx.reshape(-1)
        occ = self.elem_occupancy.reshape(-1)

        num_elems = sample_idx.shape[0]

        elem_density = np.full((num_elems,), rho_void, dtype=np.float32)
        elem_fiber = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (num_elems, 1))

        valid = (occ > 0) & (sample_idx >= 0)

        elem_density[valid] = rho_surface[sample_idx[valid]].astype(np.float32)
        elem_fiber[valid] = fiber_surface[sample_idx[valid]].astype(np.float32)

        norms = np.linalg.norm(elem_fiber[valid], axis=1, keepdims=True)
        elem_fiber[valid] /= np.clip(norms, 1e-12, None)

        self.elem_density = elem_density
        self.elem_fiber = elem_fiber
        self.elem_phi, self.elem_theta = self.fiber_vectors_to_angles(elem_fiber)

        return elem_density, elem_fiber
    
    def fiber_vectors_to_angles(self, fiber_vec):
        fiber_vec = self.to_numpy(fiber_vec).reshape(-1, 3)

        norms = np.linalg.norm(fiber_vec, axis=1, keepdims=True)
        v = fiber_vec / np.clip(norms, 1e-12, None)

        ax = v[:, 0]
        ay = v[:, 1]
        az = v[:, 2]

        phi = np.arctan2(ay, ax).astype(np.float32)
        theta = np.arccos(np.clip(az, -1.0, 1.0)).astype(np.float32)

        return phi, theta
    def assign_decoder_fields(self, rho_surface, fiber_surface, rho_void=1e-3):
        elem_density, elem_fiber = self.assign_surface_fields_to_voxels(
            rho_surface=rho_surface,
            fiber_surface=fiber_surface,
            rho_void=rho_void
        )
        elem_phi, elem_theta = self.fiber_vectors_to_angles(elem_fiber)

        self.elem_density = elem_density
        self.elem_fiber = elem_fiber
        self.elem_phi = elem_phi
        self.elem_theta = elem_theta

        return elem_density, elem_phi, elem_theta
    def show_voxels_and_surface(self):
        occ = self.elem_occupancy
        centers = self.elem_centers
        surface = self.points_xyz
        h = self.voxel_size

        vox_pts = centers[occ.astype(bool)]

        if vox_pts.shape[0] == 0:
            print("No occupied voxels to display")
            return

        plotter = pv.Plotter()

        cubes = [pv.Cube(center=p, x_length=h, y_length=h, z_length=h) for p in vox_pts]

        mesh = cubes[0]
        for c in cubes[1:]:
            mesh = mesh.merge(c)

        plotter.add_mesh(mesh, color="lightblue", opacity=0.6)

        cloud = pv.PolyData(surface)
        plotter.add_mesh(cloud, color="red", point_size=10, render_points_as_spheres=True)

        plotter.show()

    def get_flat_node_coords(self):
        """
        Returns
        -------
        node_ids : ndarray, shape (num_nodes,)
            Flat global node ids: 0, 1, 2, ..., num_nodes-1

        coords : ndarray, shape (num_nodes, 3)
            Flat node coordinates [x, y, z] for each node id.
        """
        coords = self.node_coords.reshape(-1, 3)
        node_ids = np.arange(coords.shape[0], dtype=np.int64)
        return node_ids, coords    
    def node_ids_to_dofs(self, node_ids, components=(0, 1, 2)):
        """
        Convert node ids to global DOF ids.

        Parameters
        ----------
        node_ids : array-like
            Global node ids.
        components : tuple
            Which displacement components to include:
            0 -> ux, 1 -> uy, 2 -> uz

        Returns
        -------
        dofs : ndarray
            Flat array of global DOF ids.
        """
        node_ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)

        dofs = []
        for c in components:
            dofs.append(3 * node_ids + int(c))

        if len(dofs) == 0:
            return np.array([], dtype=np.int64)

        return np.concatenate(dofs).astype(np.int64)
    
    def make_empty_force(self):
        """
        Create an empty global force vector of shape (ndof, 1).
        """
        ndof = 3 * (self.mesh['nelx'] + 1) * (self.mesh['nely'] + 1) * (self.mesh['nelz'] + 1)
        return np.zeros((ndof, 1), dtype=float)
    
    def apply_nodal_force(self, force, node_ids, direction, total_value):
        node_ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)
        if node_ids.size == 0:
            raise ValueError("No nodes selected for force application")

        comp_map = {'x': 0, 'y': 1, 'z': 2}
        c = comp_map[direction]

        val_per_node = total_value / node_ids.size
        dofs = 3 * node_ids + c
        force[dofs, 0] += val_per_node
        return force
    def set_boundary_conditions_from_regions(self, fixed_nodes, force_nodes, force_direction='z', force_value=-1.0):
        ndof = 3 * (self.mesh['nelx'] + 1) * (self.mesh['nely'] + 1) * (self.mesh['nelz'] + 1)
        force = np.zeros((ndof, 1), dtype=float)

        fixed = self.node_ids_to_dofs(fixed_nodes, components=(0, 1, 2))
        force = self.apply_nodal_force(force, force_nodes, force_direction, force_value)

        self.boundaryCondition = {
            'exampleName': self.name,
            'physics': 'Structural',
            'force': force,
            'fixed': fixed,
            'numDOFPerNode': 3
        }

    def show_voxels_surface_and_bc(self):
        occ = self.elem_occupancy
        centers = self.elem_centers
        surface = self.points_xyz
        h = self.voxel_size

        vox_pts = centers[occ.astype(bool)]

        plotter = pv.Plotter()

        if vox_pts.shape[0] > 0:
            cubes = [pv.Cube(center=p, x_length=h, y_length=h, z_length=h) for p in vox_pts]
            mesh = cubes[0]
            for c in cubes[1:]:
                mesh = mesh.merge(c)
            plotter.add_mesh(mesh, color="lightblue", opacity=0.35)

        if surface is not None and surface.shape[0] > 0:
            cloud = pv.PolyData(surface)
            plotter.add_mesh(cloud, color="red", point_size=6, render_points_as_spheres=True)

        node_ids, coords = self.get_flat_node_coords()

        fixed_dofs = self.boundaryCondition['fixed']
        fixed_node_ids = np.unique(fixed_dofs // 3)

        force = self.boundaryCondition['force'].reshape(-1)
        force_node_ids = np.unique(np.where(np.abs(force) > 0)[0] // 3)

        if fixed_node_ids.size > 0:
            fixed_pts = coords[fixed_node_ids]
            plotter.add_mesh(
                pv.PolyData(fixed_pts),
                color="green",
                point_size=12,
                render_points_as_spheres=True
            )

        if force_node_ids.size > 0:
            force_pts = coords[force_node_ids]
            plotter.add_mesh(
                pv.PolyData(force_pts),
                color="yellow",
                point_size=12,
                render_points_as_spheres=True
            )

        plotter.show_axes()

        plotter.show()    

    def occupied_axis_bounds(self):
        occ = self.elem_occupancy.astype(bool)
        if not np.any(occ):
            return self.padded_bbox_from_midsurface(self.brep_bbox, self.thickness, self.voxel_size, self.extra_layers)

        occ_centers = self.elem_centers[occ]
        half = 0.5 * self.voxel_size
        return {
            'xmin': float(np.min(occ_centers[:, 0]) - half),
            'xmax': float(np.max(occ_centers[:, 0]) + half),
            'ymin': float(np.min(occ_centers[:, 1]) - half),
            'ymax': float(np.max(occ_centers[:, 1]) + half),
            'zmin': float(np.min(occ_centers[:, 2]) - half),
            'zmax': float(np.max(occ_centers[:, 2]) + half),
        }
    
    def select_nodes_in_box(self, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None):
        """
        Select node ids whose coordinates lie inside a rectangular box.

        Any bound can be left as None to mean 'no restriction' in that direction.

        Returns
        -------
        node_ids : ndarray
            Flat array of selected global node ids.
        """
        node_ids, coords = self.get_flat_node_coords()

        mask = np.ones(coords.shape[0], dtype=bool)

        if xmin is not None:
            mask &= coords[:, 0] >= xmin
        if xmax is not None:
            mask &= coords[:, 0] <= xmax

        if ymin is not None:
            mask &= coords[:, 1] >= ymin
        if ymax is not None:
            mask &= coords[:, 1] <= ymax

        if zmin is not None:
            mask &= coords[:, 2] >= zmin
        if zmax is not None:
            mask &= coords[:, 2] <= zmax

        return node_ids[mask]
    def debug_voxel_stats(self):
        if self.elem_occupancy is None:
            print("elem_occupancy is None")
            return

        occ = self.elem_occupancy
        num_occ = int(occ.sum())
        num_total = int(occ.size)

        hx, hy, hz = self.mesh['elemSize']
        voxel_vol = hx * hy * hz
        vox_vol = num_occ * voxel_vol

        try:
            target_vol = float(np.sum(self.face_areas)) * self.thickness
        except Exception:
            target_vol = None

        print("=== Voxel Stats ===")
        print("brep_bbox:", self.brep_bbox)
        print("mesh:", self.mesh)
        print("elem_centers shape:", self.elem_centers.shape)
        print("node_coords shape:", self.node_coords.shape)
        print("occupied voxels:", num_occ)
        print("total voxels:", num_total)
        print("occupancy ratio:", num_occ / max(num_total, 1))
        print("voxelized volume:", vox_vol)
        print("thickness:", self.thickness)
        print("voxel_size:", self.voxel_size)

        if target_vol is not None:
            print("target approx volume (sum(face_areas)*thickness):", target_vol)
            if target_vol > 0:
                print("volume ratio voxel/target:", vox_vol / target_vol)
    def build_fem_fields_from_decoder(self, rho_surface, fiber_surface, rho_void=1e-3):
        elem_density, elem_phi, elem_theta = self.assign_decoder_fields(
            rho_surface=rho_surface,
            fiber_surface=fiber_surface,
            rho_void=rho_void
        )

        return {
            'density': elem_density,
            'phi': elem_phi,
            'theta': elem_theta,
            'fixed': self.boundaryCondition['fixed'],
            'force': self.boundaryCondition['force'],
            'mesh': self.mesh,
            'materialProperty': self.materialProperty,
        }
    def build_fem_fields_from_decoder_torch(self, rho_surface, fiber_surface, rho_void=1e-3):
        device = rho_surface.device

        sample_idx = torch.as_tensor(self.elem_sample_idx.reshape(-1), device=device, dtype=torch.long)
        occ = torch.as_tensor(self.elem_occupancy.reshape(-1), device=device, dtype=torch.bool)

        num_elems = sample_idx.numel()

        density = torch.full((num_elems,), rho_void, dtype=rho_surface.dtype, device=device)
        fiber = torch.zeros((num_elems, 3), dtype=fiber_surface.dtype, device=device)
        fiber[:, 0] = 1.0

        valid = occ & (sample_idx >= 0)

        density[valid] = rho_surface[sample_idx[valid]]
        fiber[valid] = fiber_surface[sample_idx[valid]]

        fiber_norm = torch.linalg.norm(fiber, dim=1, keepdim=True).clamp_min(1e-12)
        fiber = fiber / fiber_norm

        ax = fiber[:, 0]
        ay = fiber[:, 1]
        az = fiber[:, 2]

        phi = torch.atan2(ay, ax)
        theta = torch.acos(torch.clamp(az, -1.0, 1.0))

        return {
            "density": density,
            "phi": phi,
            "theta": theta,
            "fixed": self.boundaryCondition["fixed"],
            "force": self.boundaryCondition["force"],
            "mesh": self.mesh,
            "materialProperty": self.materialProperty,
        }
    def show_voxels_surface_and_bc_NEW(self):
        occ = self.elem_occupancy
        centers = self.elem_centers
        surface = self.points_xyz

        vox_pts = centers[occ.astype(bool)]

        plotter = pv.Plotter()

        if vox_pts.shape[0] > 0:
            plotter.add_mesh(
                pv.PolyData(vox_pts),
                color="lightblue",
                point_size=6,
                render_points_as_spheres=True,
            )

        if surface is not None and surface.shape[0] > 0:
            plotter.add_mesh(
                pv.PolyData(surface),
                color="red",
                point_size=4,
                render_points_as_spheres=True,
            )

        node_ids, coords = self.get_flat_node_coords()

        fixed_dofs = self.boundaryCondition['fixed']
        fixed_node_ids = np.unique(fixed_dofs // 3)

        force = self.boundaryCondition['force'].reshape(-1)
        force_node_ids = np.unique(np.where(np.abs(force) > 0)[0] // 3)

        if fixed_node_ids.size > 0:
            fixed_pts = coords[fixed_node_ids]
            plotter.add_mesh(
                pv.PolyData(fixed_pts),
                color="green",
                point_size=12,
                render_points_as_spheres=True
            )

        if force_node_ids.size > 0:
            force_pts = coords[force_node_ids]
            plotter.add_mesh(
                pv.PolyData(force_pts),
                color="yellow",
                point_size=12,
                render_points_as_spheres=True
            )

    # Better text placement
        plotter.add_text("Yellow: Applied load", position="upper_right", font_size=12, color="yellow")
        plotter.add_text("Green: Fixed nodes", position="upper_left", font_size=12, color="green")
        plotter.add_text("Blue: Occupied voxels", position="lower_left", font_size=12, color="lightblue")
        plotter.add_text("Red: Surface points", position="lower_right", font_size=12, color="red")

   
        plotter.show_axes()
        plotter.show()

#if __name__ == '__main__':
    # expects `tensors` to already exist in the current scope
    # shell_problem = ThickenShell(
    #     thickness=2.0,
    #     voxel_size=1.0,
    #     extra_layers=1,
    #     tensors=tensors
    # )

    # shell_problem.debug_voxel_stats()

    # savePath = os.path.join('data', 'settings', '{}.npy'.format(shell_problem.name))
    # shell_problem.serialize(savePath)
    # print("saved to:", savePath)
