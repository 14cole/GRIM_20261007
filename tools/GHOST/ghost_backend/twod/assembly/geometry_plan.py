"""Explicit geometry preparation for repeated queries on an immutable mesh."""
import numpy as np


class AssemblyGeometry:
    def __init__(self, mesh):
        self.mesh = mesh
        self.elements = tuple(mesh.elements)
        self.centers = np.asarray([e.center for e in self.elements], float).reshape(-1, 2)
        self.lengths = np.asarray([e.length for e in self.elements], float)
        width = len(self.elements[0].node_ids) if self.elements else 2
        self.node_ids = np.asarray([e.node_ids for e in self.elements], int).reshape(-1, width)
        self.p0 = np.asarray([e.p0 for e in self.elements], float).reshape(-1, 2)
        self.segments = np.asarray([e.p1-e.p0 for e in self.elements], float).reshape(-1, 2)
        self.normals = np.asarray([e.normal for e in self.elements], float).reshape(-1, 2)
        if self.elements:
            points = np.asarray([p for e in self.elements for p in (e.p0, e.p1)])
            self.domain_upper = float(np.linalg.norm(np.ptp(points, axis=0)))*(1+1e-12)+1e-12
        else:
            self.domain_upper = 0.
        for array in (self.centers, self.lengths, self.node_ids, self.p0, self.segments, self.normals):
            array.flags.writeable = False

    def elements_touching(self, *node_sets):
        """Element mask of every element with a node in any of the node-id sets.

        A node-to-element incidence built once makes this proportional to the
        query, not to the mesh, for repeated compressed tile queries.
        """
        if not hasattr(self, '_incidence'):
            flat = self.node_ids.reshape(-1)
            order = np.argsort(flat, kind='stable')
            counts = np.bincount(flat, minlength=len(self.mesh.nodes))
            self._incidence = (np.r_[0, np.cumsum(counts)], order // max(1, self.node_ids.shape[1]))
        starts, elements = self._incidence
        mask = np.zeros(len(self.elements), bool)
        for ids in node_sets:
            ids = np.asarray(ids, dtype=np.int64)
            if not len(ids):
                continue
            lo, hi = starts[ids], starts[ids + 1]
            lengths = hi - lo
            total = int(lengths.sum())
            if total:
                positions = np.repeat(lo - np.r_[0, np.cumsum(lengths)[:-1]], lengths) + np.arange(total)
                mask[elements[positions]] = True
        return mask

    def validate(self, mesh):
        if mesh is not self.mesh or len(mesh.elements) != len(self.elements):
            raise ValueError('Prepared assembly geometry belongs to another mesh.')
