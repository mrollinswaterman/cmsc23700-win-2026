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

"""
Smoothing Logic from here: 
    - https://nosferalatu.com/LaplacianMeshSmoothing.html
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
            # create a dictionary to hold the new position for each vertex
            # this is necessary so that subsequent vertices aren't smoothed according to the
            # smoothed position of previous vertices
            smoothed = {}
            for v in self.mesh.topology.vertices:
                # create an "empty" coordinate to hold the aggregate coordinates of the neighbors
                avg_pos = np.array([0.0, 0.0, 0.0])
                v = self.mesh.topology.vertices[v]
                neighbors = v.adjacentVertices()

                for n in neighbors:
                    # fill the empty coordinate with each neighbor coordinate, ignoring the vertex's own coordinate
                    if n.index != v.index:
                        avg_pos += self.mesh.get_3d_pos(n)

                avg_pos = avg_pos / (len(neighbors) - 1)

                smoothed[v.index] = avg_pos

            # change all vertex positions according to their entry in the dict
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

    # for debugging purposes
    # flag can be used to tell the do_collapse method to stop and view the mesh
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

    # get incident faces via the edge's halfedge and its twin
    prep.del_faces = [e.halfedge.face, e.halfedge.twin.face]
    # get one edge from each incident face, also via halfedge and twin
    prep.del_edges = [e, e.halfedge.prev().edge, e.halfedge.twin.next.edge]
    # add one vertex to del list
    prep.del_verts = [prep.merge_verts[0]]
    # for each edge to be delete, add its halfedge to the delete list
    prep.del_hes = [entry.halfedge for entry in prep.del_edges]
    # get the twin of each halfedge marked for deletion
    prep.del_hes += [entry.twin for entry in prep.del_hes]

    # check we are deleting the right amount of objects
    assert len(prep.del_hes) == 6
    assert len(prep.del_faces) == 2
    assert len(prep.del_edges) == 3

    # set the halfedge of each face to be the prev() of the halfedge being deleted
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
    # go through all the faces who will be losing a half edge, ignoring those marked for deletion
    for f in affected_faces:
        my_hes = [f.halfedge, f.halfedge.next, f.halfedge.prev()]
        my_hes += [he.twin for he in my_hes]
        verts = [he.vertex for he in my_hes]
        new_next = None
        for he in hes_to_be_modified:
            # find the halfedge who's vertex is being moved AND who already points to one of the face's vertices
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
        # for each halfege whos vertex is being moved, add it to the list
        # now pointing to the merged vertex that is not being deleted
        prep.repair_he_verts.append((i, prep.merge_verts[1]))

    # for each face that will be losing a halfedge, repair it with the new half edge found above
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

    new_vertex_pos = (
        mesh.get_3d_pos(prep.merge_verts[0]) + mesh.get_3d_pos(prep.merge_verts[1])
    ) / 2

    if prep.flag:
        mesh.view_with_topology(
            highlight_halfedges=prep.test_he,
            highlight_vertices=prep.merge_verts,
            highlight_faces=prep.del_faces,
            highlight_edges=[prep.edge],
        )

    # move the position of the vertex being kept to it's new place
    mesh.vertices[prep.merge_verts[1].index] = new_vertex_pos

    # for every halfedge that was attached to the deleted vertex, move it's vertex to the new average vertex
    for he in prep.repair_he_verts:
        mesh.topology.halfedges[he[0].index].vertex = he[1]

    for i in prep.repair_face_hes:
        f = i[0]
        if f not in mesh.topology.faces.values():
            print(f"face {f} not found in mesh topology!")
            continue
        new_he = i[1]
        if not new_he:
            print(prep)
            print(f)
            continue
        # find an appropriate halfedge of the face
        he = f.halfedge
        if he in prep.del_hes:
            f.halfedge = he.prev()
            he = he.prev()
        # determine if you need the twin of the new halfedge to maintain the
        # directionality of the face
        if he.prev().vertex == new_he.vertex:
            # print("twin needed!")
            new_he = new_he.twin

        # set the properties of the chosen new halfedge so that it is associated with the face
        # this new he replaces one deleted by the collapse
        new_he.face = f
        new_he.next = he.prev()
        he.next = new_he
        he.prev().next = he

        if prep.flag:
            print(he.next.next == he.prev(), he.prev().next == he)
            mesh.view_with_topology(highlight_halfedges=f.adjacentHalfedges())

    if prep.flag:
        mesh.view_with_topology(
            highlight_halfedges=prep.test_he, highlight_vertices=prep.merge_verts
        )

    for e in prep.del_edges:
        del mesh.topology.edges[e.index]

    for he in mesh.topology.halfedges.values():
        # if we find any halfesges still attached to the deleted vertex
        # attach them to the undeleted vertex
        if he not in prep.del_hes and he.vertex == prep.del_verts[0]:
            he.vertex = prep.merge_verts[1]
            # raise Exception(f"deleted vertex still referenced by halfedge {he}")

    del mesh.topology.vertices[prep.del_verts[0].index]

    for f in prep.del_faces:
        # print(f"deleting face at index {f.index}")
        hes = [f.halfedge, f.halfedge.next, f.halfedge.prev()]
        # check all the face's halfedges to ensure none of them still reference the
        # face before you delete it.
        # if they do, try and fix that using the halfegde's neighbors
        for he in hes:
            if he.face == f:
                # print(f"deleted face still referenced by halfedge {he}. Fixing...")
                if he.next.face != f:
                    he.face = he.next.face
                elif he.prev().face != f:
                    he.face = he.prev().face
        del mesh.topology.faces[f.index]

    for he in prep.del_hes:
        # print(f"deleting he at index {he.index}")
        # if we find that a valid halfedge still refernces a deleted halfegde
        # try to repair it using it's associated face
        if he.prev().next == he:
            for i in prep.repair_face_hes:
                f = i[0]
                if he.prev().face == f:
                    he.prev().next = i[1]
                    break

            if he.prev().next == he:
                he.prev().next = he.next
        # this should never be true because the above repair resolves it, used for debugging purposes
        if he.next.prev() == he and he.next not in prep.del_hes:
            # print(prep.edge)
            # mesh.view_with_topology(highlight_halfedges=[he, he.next.twin, he.twin.next])
            # raise Exception("deleted halfedge still connected to next")
            pass

        # should also never be true since edges are deleted along with their halfedges
        # debugging purposes
        if he.edge in mesh.topology.edges.values():
            raise Exception("deleted he still has a valid edge")

        # if we find that the halfedge's vertex still references it before deletion:
        # to to fix that by picking a different halfedge for the vertex to reference
        if he.vertex.halfedge == he and he.vertex in mesh.topology.vertices.values():
            # print("deleted halfedge still referenced by a vertex! Fixing...")
            if he.prev() not in prep.del_hes:
                he.vertex.halfedge = he.prev().twin
            else:
                adj = he.vertex.adjacentHalfedges()
                for candidate in adj:
                    if candidate != he:
                        he.vertex.halfedge = candidate
                        break

        he.face = None
        # he.next = None
        del mesh.topology.halfedges[he.index]

    # check for no holes
    assert 2 == len(mesh.topology.vertices) - len(mesh.topology.edges) + len(
        mesh.topology.faces
    )

    if not mesh.topology.consistency_check():
        print("consistency check failed!")
        return False

    # print("all good!")
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
