from typing import Optional
from dataclasses import dataclass, field
from . import Halfedge, Edge, Vertex, Face, Topology, Mesh
import numpy as np


"""
TODO complete these functions
P5 -- LaplacianSmoothing.apply
P6 -- prepare_collapse, do_collapse
Extra credit -- link_condition
"""


class MeshEdit:
    """
    Abstract interface for a mesh edit. The edit is prepared upon init
    (creating/storing info about the edit before actually executing it) then, if
    determined to be doable, applied with apply().
    """

    def __init__(self):
        pass

    def apply(self):
        pass


class LaplacianSmoothing(MeshEdit):
    def __init__(self, mesh: Mesh, n_iter: int):
        self.mesh = mesh
        self.n_iter = n_iter

    def apply(self):
        # TODO: P5 -- complete this function
        for _ in range(0, self.n_iter):
            smoothed = {}
            for v in self.mesh.topology.vertices:
                avg_pos = np.array([0.0, 0.0, 0.0])
                v = self.mesh.topology.vertices[v]
                neighbors = v.adjacentVertices()

                for n in neighbors:
                    if n.index != v.index:
                        avg_pos += self.mesh.get_3d_pos(n)

                avg_pos = avg_pos / (len(neighbors) - 1)

                smoothed[v.index] = avg_pos
                # self.mesh.vertices[v.index] = avg_pos

            for i in smoothed.keys():
                self.mesh.vertices[i] = smoothed[i]


class EdgeCollapse(MeshEdit):
    def __init__(self, mesh: Mesh, e_id: int):
        self.mesh = mesh
        self.e_id = e_id
        self.prep = prepare_collapse(self.mesh, self.e_id)

    def apply(self):
        return do_collapse(self.prep, self.mesh)


@dataclass
class CollapsePrep:
    """
    A data-class that stores all the operations you may need to perform during
    an edge collapse.

    The intention of this data-class is to help keep everything organized and
    remind you of what aspects of the half-edge mesh structure you need to keep
    track of. Depending on your implementation, you very likely will not need to
    use all of the parameters below.

    Read this link to learn more about data-classes in Python:
    https://www.dataquest.io/blog/how-to-use-python-data-classes/
    """

    # The vertices that need to be merged through the edge collapse
    merge_verts: tuple[Vertex, Vertex]

    # The primitives that need their references updated. Each list item is a
    # tuple (primitive_that_needs_a_reference_fix, new_primitive_it_should_point_to)
    #
    # The field(default_factory=list) default value just means that each field will be
    # initialized as an empty list which is instantiated when this dataclass is instantiated
    repair_he_twins: list[tuple[Halfedge, Halfedge]] = field(default_factory=list)
    repair_he_nexts: list[tuple[Halfedge, Halfedge]] = field(default_factory=list)
    repair_he_edges: list[tuple[Halfedge, Edge]] = field(default_factory=list)
    repair_he_verts: list[tuple[Halfedge, Vertex]] = field(default_factory=list)
    repair_he_faces: list[tuple[Halfedge, Face]] = field(default_factory=list)
    repair_edge_hes: list[tuple[Edge, Halfedge]] = field(default_factory=list)
    repair_vert_hes: list[tuple[Vertex, Halfedge]] = field(default_factory=list)
    repair_face_hes: list[tuple[Face, Halfedge]] = field(default_factory=list)

    # The primitives that need to be deleted
    del_verts: list[Vertex] = field(default_factory=list)
    del_edges: list[Edge] = field(default_factory=list)
    del_hes: list[Halfedge] = field(default_factory=list)
    del_faces: list[Face] = field(default_factory=list)

    test_he: list = field(default_factory=list)
    test_face: list = field(default_factory=list)
    test_v: list = field(default_factory=list)

    flag: bool = False
    edge: Edge = None


# TODO: P6 -- complete this
def prepare_collapse(mesh: Mesh, e_id: int) -> CollapsePrep:
    """
    The first stage of edge-collapse.

    This function should traverse the mesh's topology and figure out which
    operations are needed to perform an edge collapse. These operations should
    be stored in a `CollapsePrep` object (see definition above) that is returned.
    """
    topology = mesh.topology
    e = topology.edges[e_id]

    prep = CollapsePrep(e.two_vertices())

    prep.edge = e

    # prep.merge_verts =

    # get incident faces via the edge's halfedge and its twin
    prep.del_faces = [e.halfedge.face, e.halfedge.twin.face]
    # get one edge from each incident face, also via halfedge and twin
    prep.del_edges = [e, e.halfedge.prev().edge, e.halfedge.twin.next.edge]
    # add one vertex to del list
    prep.del_verts = [prep.merge_verts[0]]
    # for each edge to be delete, add its halfedge to the delete list
    prep.del_hes = [entry.halfedge for entry in prep.del_edges]
    prep.del_hes += [entry.twin for entry in prep.del_hes]

    # set the halfedge of each face to be prev() of the halfedge being deleted
    # this is so a new half edge can be easily slotted in later
    for he in prep.del_hes:
        he.face.halfedge = he.prev()

    # find the halfedges that will be edited, ignoring those marked for deletion

    hes_to_be_modified = [
        i
        for i in prep.merge_verts[1].adjacentHalfedges()
        if i.vertex == prep.merge_verts[1]
        if i not in prep.del_hes
        if i.twin not in prep.del_hes
    ]
    hes_to_be_modified += [
        i
        for i in prep.merge_verts[0].adjacentHalfedges()
        if i.vertex == prep.merge_verts[0]
        if i not in prep.del_hes
        if i.twin not in prep.del_hes
    ]

    # get faces with a halfedge being deleted
    affected_faces = [he.face for he in prep.del_hes if he.face not in prep.del_faces]

    new_face_hes = []
    for f in affected_faces:
        my_hes = [f.halfedge, f.halfedge.next, f.halfedge.prev()]
        my_hes += [he.twin for he in my_hes]
        verts = [he.vertex for he in my_hes]
        new_next = None
        for he in hes_to_be_modified:
            # find the halfedge who's vertex is being moved and who points to one of the face's vertices
            # this will be the halfedge that replaces the deleted halfedge for that face
            if he not in my_hes and he.tip_vertex() in verts:
                new_next = he

        if not new_next:
            continue
            print(verts)
            for he in hes_to_be_modified:
                print(he.tip_vertex(), he.vertex)
            raise Exception

        new_face_hes.append(new_next)

    for i in hes_to_be_modified:
        # if i not in prep.del_hes and i.twin not in prep.del_hes:
        prep.repair_he_verts.append((i, prep.merge_verts[1]))

    for idx, k in enumerate(new_face_hes):
        prep.repair_face_hes.append((affected_faces[idx], k))
        prep.test_face.append(affected_faces[idx])
        prep.test_he.append(affected_faces[idx].halfedge)
        prep.test_he.append(affected_faces[idx].halfedge.next)

    prep.test_face = affected_faces
    prep.test_he = hes_to_be_modified
    mesh.topology.consistency_check()

    return prep


# TODO: P6 -- complete this
def do_collapse(prep: CollapsePrep, mesh: Mesh):
    """
    The second stage of edge-collapse.

    This function should implement all of the operations described in the
    `CollapsePrep` data-class (defined above). Ideally, this function should
    not need to traverse the mesh's topology at all, as all traversal should
    be handled by prepare_collapse().

    This should modify the mesh's topology and vertices coords inplace.
    (You should not need to create any new Primitives!)

    To delete primitives, for instance, a halfedge with index halfedge_id, use
        del mesh.topology.halfedges[halfedge_id]
    and similarly for other primitive types.
    """
    # TODO write your code here and replace this raise

    prep.test_he = []

    verts = [he.vertex for he in prep.del_hes if he.vertex not in prep.del_verts]

    new_vertex_pos = (
        mesh.get_3d_pos(prep.merge_verts[0]) + mesh.get_3d_pos(prep.merge_verts[1])
    ) / 2

    for e in prep.del_edges:
        del mesh.topology.edges[e.index]

    del mesh.topology.vertices[prep.del_verts[0].index]

    if prep.edge.index == 1901:
        mesh.view_with_topology(highlight_faces=prep.del_faces)
        #

    for he in prep.del_hes:
        # print(f"deleting he at index {he.index}")
        del mesh.topology.halfedges[he.index]

    for f in prep.del_faces:
        # print(f"deleting face at index {f.index}")
        if prep.edge.index == 1901:
            print(f)
        del mesh.topology.faces[f.index]

    mesh.vertices[prep.merge_verts[1].index] = new_vertex_pos

    for he in prep.repair_he_verts:
        mesh.topology.halfedges[he[0].index].vertex = he[1]

        # make sure to reset the halfedge of the vertex that was kept and moved from the deleted edge
        mesh.topology.vertices[prep.merge_verts[1].index].halfedge = he[0]

    if prep.edge.index == 1901:
        print(prep)
        mesh.view_with_topology(highlight_halfedges=mesh.topology.vertices[prep.merge_verts[1].index].adjacentHalfedges())
        #return False

    for i in prep.repair_face_hes:
        f = i[0]
        if f not in mesh.topology.faces.items():
            print(f"face {f} not found in mesh topology!")
            continue
        new_he = i[1]
        if not new_he:
            print(prep)
            print(f)
            continue
        he = f.halfedge
        if he.prev().vertex == new_he.vertex:
            # print("twin needed!")
            new_he = new_he.twin
        new_he.face = f
        new_he.next = he.prev()
        he.next = new_he
        he.prev().next = he

    if prep.edge.index == 1901:
        #print(prep)
        mesh.view_with_topology(highlight_halfedges=mesh.topology.vertices[prep.merge_verts[1].index].adjacentHalfedges())
        return False

    if prep.edge.index == 172:
        print(prep)
        mesh.view_with_topology(
            highlight_faces=[mesh.topology.faces[599]],
        )
        print("target edge found, ending early...")
        
        return False

    

    assert 2 == len(mesh.topology.vertices) - len(mesh.topology.edges) + len(
        mesh.topology.faces
    )

    for v in verts:
        if v.halfedge in prep.del_hes:
            v.halfedge = v.halfedge.prev()

    for he in prep.del_hes:
        #he.next = None
        #he.face = None
        pass
        # print("vertex needs adjustment")

    if not mesh.topology.consistency_check():
        print(prep.edge)
        for he in mesh.topology.halfedges.values():
            if he.next not in mesh.topology.halfedges.values():
                print(he, he.next)

        print("consistency check failed!")
        return False
    else:
        #print("passed consistency check")
        return True

    return


class EdgeCollapseWithLink(MeshEdit):
    def __init__(self, mesh: Mesh, e_id: int):
        self.mesh = mesh
        self.e_id = e_id
        self.link_cond = link_condition(self.mesh, self.e_id)
        if self.link_cond:
            self.prep = prepare_collapse(self.mesh, self.e_id)

    def apply(self):
        if not self.link_cond:
            print(f"Collapse is not doable, does not satisfy link condition")
            return
        return do_collapse(self.prep, self.mesh)


# TODO: Extra credit -- complete this
def link_condition(mesh: Mesh, e_id: int) -> bool:
    """
    Return whether the mesh and the specified edge satisfy the link condition.
    """
    topology = mesh.topology
    e = topology.edges[e_id]
    # TODO write your code here and replace this raise and return
    raise NotImplementedError("TODO (EC)")
