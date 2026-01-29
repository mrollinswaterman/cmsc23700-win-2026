from typing import Sequence, Optional
import os
import numpy as np
from PIL import Image
import gzip


class TriangleMesh:
    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        face_colors: Optional[np.ndarray] = None,
        vertex_colors: Optional[np.ndarray] = None,
    ):
        """
        vertices: (n_vertices, 3) float array of vertex positions
        faces: (n_faces, 3) int array of indices into vertices, each row the verts of a triangle
        face_colors: (n_faces, 3) float array of a rgb color (in range [0,1]) per face
        vertex_colors: (n_vertices, 3) float array of a rgb color (in range [0,1]) per vertex
        """
        self.vertices = vertices
        self.faces = faces
        self.face_colors = face_colors
        self.vertex_colors = vertex_colors


def save_image(fname: str, arr: np.ndarray) -> np.ndarray:
    """
    :param fname: path of where to save the image
    :param arr: numpy array of shape (H,W,3), and should be between 0 and 1

    saves both the image and an .npy.gz file of the original image array
    and returns back the original array
    """
    im = Image.fromarray(np.clip(np.floor(arr * 256), 0, 255).astype(np.uint8))
    im.save(fname)
    with gzip.GzipFile(os.path.splitext(fname)[0] + ".npy.gz", "w") as f:
        np.save(f, arr)
    return arr


def read_image(fname: str) -> np.ndarray:
    """reads image file and returns as numpy array (H,W,3) rgb in range [0,1]"""
    return np.asarray(Image.open(fname)).astype(np.float64) / 255


"""
The following functions (make_*, calc_*, and update_zbuffer) are merely a
suggested outline for planning out your solution. They should give you an idea
of the subtasks involved and how you might factor out common code between parts.
You are free to modify these functions and organize your code however you like.
"""


class Coordinate:

    def __init__(self, x, y, z=1.0, w=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

        self.current = 0

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z}, {self.w})"

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        match self.current:
            case 1:
                return self.x
            case 2:
                return self.y
            case _:
                raise StopIteration


def make_viewport_matrix(im_h: int, im_w: int) -> np.ndarray:
    """
    You're free to modify this function's signature; this is merely a suggested
    way to factor out a common subtask.
    """
    # wtf is a viewpoer matrix: In short, it's the transformation of numbers in the range [-1, 1]
    # to numbers corresponding to pixels on the screen, which is a linear mapping computed with linear interpolation.
    # https://www.mauriciopoppe.com/notes/computer-graphics/viewing/viewport-transform/
    # matrix: [
    #   [ (im_w*0.5), 0, 0, (im_w*0.5) ]
    #   [ 0, -(im_h * 0.5), 0, (im_h * 0.5) ],
    #   [ 0, 0, 1, 0 ],
    #   [ 0, 0, 0, 1 ]
    # ]
    return np.array(
        [
            [(im_w * 0.5), 0, 0, (im_w * 0.5)],
            [0, (-im_h * 0.5), 0, (im_h * 0.5)],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def make_orthographic_matrix(
    l: float = 0.0,
    r: float = 12.0,
    b: float = 0.0,
    t: float = 12.0,
    n: float = 12.0,
    f: float = 0.0,
) -> np.ndarray:
    """
    You're free to modify this function's signature; this is merely a suggested
    way to factor out a common subtask.

    (These default argument values are the orthographic view volume parameters
    for P2 and P3.)
    """
    return np.array(
        [
            [2 / (r - l), 0, 0, -((r + l) / (r - l))],
            [0, 2 / (t - b), 0, -((t + b) / (t - b))],
            [0, 0, 2 / (n - f), -((n + f) / (n - f))],
            [0, 0, 0, 1],
        ]
    )


def make_camera_matrix(
    eye: np.ndarray, lookat: np.ndarray, up: np.ndarray
) -> np.ndarray:
    """
    You're free to modify this function's signature; this is merely a suggested
    way to factor out a common subtask.
    """
    # e = eye
    # g = lookat
    # t = up

    # Methodology from here: https://learnopengl.com/Getting-Started/Camera and textbook

    g = np.subtract(eye, lookat)  # gaze vector

    w = g / np.linalg.norm(g)  # normalized gaze vector

    u = np.cross(up, w) / np.linalg.norm(np.cross(up, w))

    v = np.cross(w, u)

    part1 = np.array(
        [
            [u[0], u[1], u[2], 0.0],
            [v[0], v[1], v[2], 0.0],
            [w[0], w[1], w[2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    part2 = np.array(
        [
            [1.0, 0.0, 0.0, -eye[0]],
            [0.0, 1.0, 0.0, -eye[1]],
            [0.0, 0.0, 1.0, -eye[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # print(np.matmul(part1, part2))

    return np.matmul(part1, part2)


def make_perspective_matrix(
    fovy: float, aspect: float, n: float, f: float
) -> np.ndarray:
    """
    You're free to modify this function's signature; this is merely a suggested
    way to factor out a common subtask.
    """

    fovy = np.deg2rad(fovy)

    P = np.array(
        [
            [n, 0.0, 0.0, 0.0],
            [0.0, n, 0.0, 0.0],
            [0.0, 0.0, n + f, -(f * n)],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )

    t = np.tan(fovy / 2) * np.abs(n)

    right = aspect * t

    ortho = make_orthographic_matrix(r=right, l=0 - right, t=t, b=0 - t, n=n, f=f)

    return np.matmul(ortho, P)
    # return P


def area(v1: Coordinate, v2: Coordinate, v3: Coordinate):
    # Returns the area of the trinagle created by the three parameter points
    return 0.5 * (
        v1.x * v2.y
        + v2.x * v3.y
        + v3.x * v1.y
        - v1.x * v3.y
        - v2.x * v1.y
        - v3.x * v2.y
    )


def get_barycentric_coordinates(
    face_vertices: list[Coordinate | tuple[float, float]], point: Coordinate
) -> list[float]:
    v1, v2, v3 = face_vertices

    bigA = area(v1, v2, v3)

    if bigA == 0:
        bigA += 0.000001

    alpha = area(point, v2, v3) / bigA
    beta = area(v1, point, v3) / bigA
    gamma = area(v1, v2, point) / bigA

    return [alpha, beta, gamma]


def calc_coverage(
    face_vertices: list[Coordinate | tuple[float, float]], point: Coordinate
):
    """
    You're free to modify this function's signature; this is merely a suggested
    way to factor out a common subtask.
    """
    # methodology from https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Equations_in_barycentric_coordinates

    barycentric = get_barycentric_coordinates(face_vertices, point)

    if not barycentric:
        return False

    alpha, beta, gamma = barycentric

    return (
        (0 - 1e-15) <= alpha <= (1 + 1e-15)
        and (0 - 1e-15) <= beta <= (1 + 1e-15)
        and (0 - 1e-15) <= gamma <= (1 + 1e-15)
    )


class BoundingBox:

    def __init__(self, vertices: list[tuple[float | int, float | int] | Coordinate]):
        # check for mins and maxes based on paramter type
        match vertices[0]:
            case Coordinate():
                self.min = Coordinate(
                    min(v.x for v in vertices),
                    min(v.y for v in vertices),
                )
                self.max = Coordinate(
                    max(v.x for v in vertices),
                    max(v.y for v in vertices),
                )
            case _:
                self.min: Coordinate = Coordinate(
                    min(x[0] for x in vertices), min(y[1] for y in vertices)
                )
                self.max: Coordinate = Coordinate(
                    max(x[0] for x in vertices), max(y[1] for y in vertices)
                )

        self._vertices = vertices

    def to_int(self):
        self.min.x = int(np.floor(self.min.x))
        self.min.y = int(np.floor(self.min.y))
        self.max.x = int(np.ceil(self.max.x))
        self.max.y = int(np.ceil(self.max.y))

    def __str__(self) -> str:
        return (
            f"Min: ({self.min.x}, {self.min.y})  |   Max: ({self.max.x}, {self.max.y})"
        )


def calc_triangle_bounding_box(obj: TriangleMesh, face: np.ndarray):
    """
    You're free to modify this function's signature; this is merely a suggested
    way to factor out a common subtask.
    """
    _2D_vertices = [
        (float(obj.vertices[x][0]), float(obj.vertices[x][1])) for x in face
    ]

    # get bounding box
    return BoundingBox(_2D_vertices)


def NCD_2_SC(
    mvp: np.ndarray, point: tuple[int | float, int | float]
) -> tuple[int, int]:
    """
    Map Canonical / NCD Coordinates to Screen Coordinates

    Input:
        mvp = viewport maxtrix based on the image w and h
        point: point to be translated

    Returns:
        a tuple of ints, because screen should consider real numbers!
    """
    point = np.array(
        [
            point[0],
            point[1],
            1.0,
            1.0,
        ]
    )

    sc = np.matmul(mvp, point)
    return (int(sc[0]), int(sc[1]))


def SC_2_NCD(pixel: tuple[int | float, int | float], im_w, im_h) -> np.ndarray:
    # screen origin = (-1, 1) top (y = 1) and left (x = -1)
    ncd_x = (pixel[0] / im_w - 0.5) * 2
    ncd_y = -(pixel[1] / im_h - 0.5) * 2

    return (ncd_x, ncd_y)


def ortho_2_SC(point: Coordinate, matrix: np.ndarray) -> Coordinate:
    # construct a dummy matrix to match the required shape of the transformation matrix
    pixel = np.matmul(matrix, np.array([point.x, point.y, 1.0, 1.0]))
    return Coordinate(pixel[0], pixel[1])


def cam_2_SC(point: Coordinate, matrix: np.ndarray) -> Coordinate:
    # dummy matrix from ortho_2_SC needs the z-axis information now to render all faces properly
    pixel = np.matmul(matrix, np.array([point.x, point.y, point.z, 1.0]))
    return Coordinate(pixel[0], pixel[1])


def per_2_SC(point: Coordinate, matrix: np.ndarray) -> Coordinate:
    # dummy matrix from cam_2_SC now takes the w value of the point into account
    # when we retuen the screen coordinate, first do the perspective correction of
    # dividing by w
    pixel = np.matmul(matrix, np.array([point.x, point.y, point.z, point.w]))
    return Coordinate(pixel[0] / pixel[3], pixel[1] / pixel[3], pixel[2], pixel[3])


def update_zbuffer(zbuffer: np.ndarray, YOUR_OTHER_ARGUMENTS_ETC):
    """
    You're free to modify this function's signature; this is merely a suggested
    way to factor out a common subtask.
    """
    pass


"""
The functions below are the ones actually run for grading. Do not change the
signatures of these functions. The autograder will run them and expect the
result image to be returned from them.
"""


# P1
def render_viewport(obj: TriangleMesh, im_w: int, im_h: int) -> np.ndarray:
    """
    Render out just the vertices of each triangle in the input object.
    TIP: Pad the vertex pixel out in order to visualize properly like in the
    handout pdf (but turn that off when you submit your code)
    """
    img = np.zeros((im_h, im_w, 3))
    mvp = make_viewport_matrix(im_w=im_w, im_h=im_h)

    # for each face in the obj
    for idx, face in enumerate(obj.faces):
        # get it's vertices
        vertices = [obj.vertices[v] for v in face]

        # convert them to screen coords
        sc_vertices = [NCD_2_SC(mvp, (p[0], p[1])) for p in vertices]

        # draw them
        for v in sc_vertices:
            img[v[1], v[0]] = obj.face_colors[idx]

    return save_image("p1.png", img)


# P2
def render_ortho(obj: TriangleMesh, im_w: int, im_h: int) -> np.ndarray:
    """Render the orthographic projection of the cube"""
    img = np.zeros((im_h, im_w, 3))
    mvp = make_viewport_matrix(im_w=im_w, im_h=im_h)
    ortho = make_orthographic_matrix()

    bigM = np.matmul(mvp, ortho)

    for face_idx, face in enumerate(obj.faces):

        # get the vertices of the face (triangle)
        face_vertices = [
            Coordinate(obj.vertices[x][0], obj.vertices[x][1]) for x in face
        ]

        # convert them to screen coordinates
        face_vertices = [ortho_2_SC(v, bigM) for v in face_vertices]

        # get the bounding box of the triangle, in screen coordinates
        bb = BoundingBox(face_vertices)
        bb.to_int()  # turn the mins and maxes to integers so we can iterate over them

        for x in range(bb.min.x, bb.max.x + 1):
            for y in range(bb.min.y, bb.max.y + 1):
                center = Coordinate(x + 0.5, y + 0.5)

                if calc_coverage(face_vertices, center):
                    img[y, x] = obj.face_colors[face_idx]

    return save_image("p2.png", img)


# P3
def render_camera(obj: TriangleMesh, im_w: int, im_h: int) -> np.ndarray:
    """Render the orthographic projection of the cube with the specific camera settings"""
    img = np.zeros((im_h, im_w, 3))
    mvp = make_viewport_matrix(im_w=im_w, im_h=im_h)
    ortho = make_orthographic_matrix()
    cam = make_camera_matrix(
        eye=np.array([0.2, 0.2, 1.0]),
        lookat=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
    )

    smallM = np.matmul(mvp, ortho)

    bigM = np.matmul(smallM, cam)

    for face_idx, face in enumerate(obj.faces):

        # get the vertices of the face (triangle)
        face_vertices = [
            Coordinate(obj.vertices[x][0], obj.vertices[x][1], obj.vertices[x][2])
            for x in face
        ]

        # convert them to screen coordinates
        face_vertices = [cam_2_SC(v, bigM) for v in face_vertices]

        # get the bounding box of the triangle
        bb = BoundingBox(face_vertices)
        bb.to_int()

        for x in range(bb.min.x, bb.max.x + 1):
            for y in range(bb.min.y, bb.max.y + 1):
                center = Coordinate(x + 0.5, y + 0.5)
                if calc_coverage(face_vertices, center):
                    img[y, x] = obj.face_colors[face_idx]

    return save_image("p3.png", img)


# P4
def render_perspective(obj: TriangleMesh, im_w: int, im_h: int) -> np.ndarray:
    """Render the perspective projection with perspective divide"""
    n = -1
    f = -100
    img = np.zeros((im_h, im_w, 3))
    mvp = make_viewport_matrix(im_w=im_w, im_h=im_h)
    per = make_perspective_matrix(fovy=65.0, aspect=4 / 3, n=n, f=f)
    cam = make_camera_matrix(
        eye=np.array([1.0, 1.0, 1.0]),
        lookat=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
    )

    smallM = np.matmul(mvp, per)

    bigM = np.matmul(smallM, cam)

    for face_idx, face in enumerate(obj.faces):

        # get the vertices of the face (triangle)
        face_vertices = [
            Coordinate(obj.vertices[x][0], obj.vertices[x][1], obj.vertices[x][2], 1.0)
            for x in face
        ]

        # convert them to screen coordinates
        face_vertices = [per_2_SC(v, bigM) for v in face_vertices]

        # get the bounding box of the triangle
        bb = BoundingBox(face_vertices)
        bb.to_int()

        for x in range(bb.min.x, bb.max.x + 1):
            for y in range(bb.min.y, bb.max.y + 1):
                pixel = Coordinate(x + 0.5, y + 0.5)

                if calc_coverage(face_vertices, pixel):
                    img[y, x] = obj.face_colors[face_idx]

    return save_image("p4.png", img)


# P5
def render_zbuffer_with_color(obj: TriangleMesh, im_w: int, im_h: int) -> np.ndarray:
    """Render the input with z-buffering and color interpolation enabled"""
    img = np.zeros((im_h, im_w, 3))
    # create the z-buffer
    zbuffer = np.zeros((im_w, im_h, 1))
    # fill it with infinity
    zbuffer.fill(np.inf)
    mvp = make_viewport_matrix(im_w=im_w, im_h=im_h)
    per = make_perspective_matrix(fovy=65.0, aspect=4 / 3, n=-1, f=-100)
    cam = make_camera_matrix(
        eye=np.array([1.0, 1.0, 1.0]),
        lookat=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
    )

    smallM = np.matmul(mvp, per)

    bigM = np.matmul(smallM, cam)

    for face_idx, face in enumerate(obj.faces):

        # get the vertices of the face (triangle)
        face_vertices = [
            Coordinate(obj.vertices[x][0], obj.vertices[x][1], obj.vertices[x][2], 1.0)
            for x in face
        ]

        # convert them to screen coordinates
        face_vertices = [per_2_SC(v, bigM) for v in face_vertices]

        # get the bounding box of the triangle
        bb = BoundingBox(face_vertices)
        bb.to_int()

        for x in range(bb.min.x, bb.max.x + 1):
            for y in range(bb.min.y, bb.max.y + 1):
                center = Coordinate(x + 0.5, y + 0.5)

                if calc_coverage(face_vertices, center):
                    # alpha = without v1
                    # beta = without v2
                    # gamma = without v3

                    # find barycentric coords
                    alpha, beta, gamma = get_barycentric_coordinates(
                        face_vertices, Coordinate(x, y)
                    )

                    # find the color (c) at each vertex of the face
                    c1, c2, c3 = [obj.vertex_colors[i] for i in face]

                    # interpolate the color for the current pixel
                    color = np.array(c1 * alpha + c2 * beta + c3 * gamma)

                    # find the depth / z coordinate for each face
                    # (after transformation to screen coordinates)
                    d1, d2, d3 = [v.z for v in face_vertices]

                    # interpolate the depth using barycentric coords
                    depth = (d1 * alpha) + (d2 * beta) + (d3 * gamma)

                    # check the interpolated depth against the pixel's z-buffer
                    if depth + (1e-15) < zbuffer[x, y]:
                        zbuffer[x, y] = depth
                        img[y, x] = color

    return save_image("p5.png", img)


# P6
def render_big_scene(
    objlist: Sequence[TriangleMesh], im_w: int, im_h: int
) -> np.ndarray:
    """Render a big scene with multiple shapes"""
    img = np.zeros((im_h, im_w, 3))
    zbuffer = np.zeros((im_w, im_h, 1))
    zbuffer.fill(np.inf)
    mvp = make_viewport_matrix(im_w=im_w, im_h=im_h)
    per = make_perspective_matrix(fovy=65.0, aspect=4 / 3, n=-1, f=-100)
    cam = make_camera_matrix(
        eye=np.array([-0.5, 1.0, 1.0]),
        lookat=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
    )

    smallM = np.matmul(mvp, per)

    bigM = np.matmul(smallM, cam)

    # do the same as z-buffer / color interpolation but for every object in the scene
    for obj in objlist:
        for face_idx, face in enumerate(obj.faces):

            # get the vertices of the face (triangle)
            face_vertices = [
                Coordinate(
                    obj.vertices[x][0], obj.vertices[x][1], obj.vertices[x][2], 1.0
                )
                for x in face
            ]

            # convert them to screen coordinates
            face_vertices = [per_2_SC(v, bigM) for v in face_vertices]

            # get the bounding box of the triangle
            bb = BoundingBox(face_vertices)
            bb.to_int()

            for x in range(bb.min.x, bb.max.x + 1):
                for y in range(bb.min.y, bb.max.y + 1):
                    center = Coordinate(x + 0.5, y + 0.5)

                    if calc_coverage(face_vertices, center):
                        # alpha = without v1
                        # beta = without v2
                        # gamma = without v3
                        alpha, beta, gamma = get_barycentric_coordinates(
                            face_vertices, Coordinate(x, y)
                        )

                        # get the colors for each vertex of the face
                        c1, c2, c3 = [obj.vertex_colors[i] for i in face]

                        # use those vertex colors to interpolate color for the current pixel
                        color = np.array(c1 * alpha + c2 * beta + c3 * gamma)

                        # depth interpolation again (see render_z-buffer)

                        d1, d2, d3 = [v.z for v in face_vertices]

                        depth = (d1 * alpha) + (d2 * beta) + (d3 * gamma)

                        if depth < zbuffer[x, y]:
                            zbuffer[x, y] = depth
                            img[y, x] = color

    return save_image("p6.png", img)


# P7
def my_cube_uvs(cube: TriangleMesh) -> np.ndarray:
    """
    Build your own UV coordinates for the cube mesh to test out texture_map.
    You may choose to hard-code numbers or compute a planar parameterization of
    the cube mesh. The UVs should have shape (n_faces, 3, 2), i.e. UV coordinates
    (u,v) for each of the 3 corners of each face.

    Note that this function is for you to use to make input UVs for running your
    implementation of texture_map, for reproducing p7.png, and for creating your
    custom.png. The autograder will not be running this. The autograder will
    call texture_map with our UVs.
    """
    # You may find it helpful to start with this array and fill it out with correct values.
    uvs = np.zeros((len(cube.faces), 3, 2))
    i = 0

    # take each face in pairs of two, with each pair representing a full "side" of the cube
    # then give each vertex of that face pairing a coordinate in u,v space
    # finally, move to the next pairing and repeat
    while i < len(cube.faces) - 1:
        face1 = cube.faces[i]
        face2 = cube.faces[i + 1]

        uvs[i][0] = [1, 1]  # u,v of vertex 1
        uvs[i][1] = [0, 1]  # u,v of vertex 2
        uvs[i][2] = [1, 0]  # u,v of vertex 3

        uvs[i + 1][0] = [1, 0]  # u,v of vertex 1
        uvs[i + 1][1] = [0, 1]  # u,v of vertex 2
        uvs[i + 1][2] = [0, 0]  # u,v of vertex 3

        i = i + 2
    return uvs


def texture_map(
    obj: TriangleMesh, uvs: np.ndarray, img: np.ndarray, im_w: int, im_h: int
) -> np.ndarray:
    """
    Render a cube with the texture img mapped onto its faces according to uvs.
    `uvs` has shape (n_faces, 3, 2) and contains the UV coordinates (u,v) for
    each of the 3 corners of each face
    `img` has shape (height, width, 3)
    """
    t_w = t_h = len(img) - 1
    zbuffer = np.zeros((im_w, im_h, 1))
    zbuffer.fill(np.inf)
    output = np.zeros((im_h, im_w, 3))

    mvp = make_viewport_matrix(im_w=im_w, im_h=im_h)
    per = make_perspective_matrix(fovy=65.0, aspect=4 / 3, n=-1, f=-100)
    cam = make_camera_matrix(
        eye=np.array([1.0, 1.0, 1.0]),
        lookat=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
    )

    smallM = np.matmul(mvp, per)

    bigM = np.matmul(smallM, cam)

    for face_idx, face in enumerate(obj.faces):

        # get the vertices of the face (triangle)
        face_vertices = [
            Coordinate(obj.vertices[x][0], obj.vertices[x][1], obj.vertices[x][2], 1.0)
            for x in face
        ]

        # convert them to screen coordinates
        face_vertices = [per_2_SC(v, bigM) for v in face_vertices]

        # get the bounding box of the triangle
        bb = BoundingBox(face_vertices)
        bb.to_int()

        for x in range(bb.min.x, bb.max.x + 1):
            for y in range(bb.min.y, bb.max.y + 1):

                center = Coordinate(x + 0.5, y + 0.5)

                if calc_coverage(face_vertices, center):
                    # alpha = without v1
                    # beta = without v2
                    # gamma = without v3
                    alpha, beta, gamma = get_barycentric_coordinates(
                        face_vertices, Coordinate(x, y)
                    )

                    # u = bary * (u_0 / w_0) + bary * (u_1 / w_1) + bary * (u_2 / w_2)

                    # get the u,v coordinates for each vertex of the face
                    uvVertices = [uvs[face_idx][i] for i in range(len(face))]

                    # calcuate uS, vS and 1S as per the slides from lecture 05
                    uS = (
                        alpha * (uvVertices[0][0] / face_vertices[0].w)
                        + beta * (uvVertices[1][0] / face_vertices[1].w)
                        + gamma * (uvVertices[2][0] / face_vertices[2].w)
                    )

                    vS = (
                        alpha * (uvVertices[0][1] / face_vertices[0].w)
                        + beta * (uvVertices[1][1] / face_vertices[1].w)
                        + gamma * (uvVertices[2][1] / face_vertices[2].w)
                    )

                    _1S = (
                        alpha * (1 / face_vertices[0].w)
                        + beta * (1 / face_vertices[1].w)
                        + gamma * (1 / face_vertices[2].w)
                    )

                    # use the above calculations to find u and v proper
                    # (again as per lecture 5)
                    u = uS / _1S
                    v = vS / _1S

                    # clamp as recommended in the textbook

                    i = round(u * t_w - 0.5) % t_w
                    j = round(v * t_h - 0.5) % t_h

                    color = img[j, i]

                    # depth interpolation remains the same

                    d1, d2, d3 = [v.z for v in face_vertices]

                    depth = (d1 * alpha) + (d2 * beta) + (d3 * gamma)

                    if depth < zbuffer[x, y]:
                        zbuffer[x, y] = depth
                        output[y, x] = color

    return save_image("p7.png", output)


####### setup stuff
def get_big_scene():
    # Cube
    vertices = np.array(
        [
            [-0.35, -0.35, -0.15],
            [-0.15, -0.35, -0.15],
            [-0.35, -0.15, -0.15],
            [-0.15, -0.15, -0.15],
            [-0.35, -0.35, -0.35],
            [-0.15, -0.35, -0.35],
            [-0.35, -0.15, -0.35],
            [-0.15, -0.15, -0.35],
        ]
    )
    triangles = np.array(
        [
            [0, 1, 2],
            [2, 1, 3],
            [5, 4, 7],
            [7, 4, 6],
            [4, 0, 6],
            [6, 0, 2],
            [1, 5, 3],
            [3, 5, 7],
            [2, 3, 6],
            [6, 3, 7],
            [4, 5, 0],
            [0, 5, 1],
        ]
    )
    vertex_colors = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    cube1 = TriangleMesh(vertices, triangles, vertex_colors=vertex_colors)

    # Tet
    vertices = np.array(
        [[0.45, 0.5, 0.35], [0.4, 0.4, 0.45], [0.4, 0.35, 0.25], [0.4, 0.45, 0.3]]
    )
    triangles = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    vertex_colors = np.array(
        [[1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]
    )
    tet1 = TriangleMesh(vertices, triangles, vertex_colors=vertex_colors)

    # Tet
    vertices = np.array(
        [[0.0, 0.0, 0.0], [-0.1, -0.3, -0.25], [-0.1, 0.1, 0.3], [-0.1, -0.15, 0.4]]
    )
    triangles = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    vertex_colors = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0]]
    )
    tet2 = TriangleMesh(vertices, triangles, vertex_colors=vertex_colors)

    vertices = np.array(
        [
            [-0.4, -0.4, 0.2],
            [-0.5, -0.4, 0.2],
            [-0.4, -0.5, 0.2],
            [-0.5, -0.5, 0.2],
            [-0.4, -0.4, 0.3],
            [-0.5, -0.4, 0.3],
            [-0.4, -0.5, 0.3],
            [-0.5, -0.5, 0.3],
        ]
    )
    triangles = np.array(
        [
            [0, 1, 2],
            [2, 1, 3],
            [5, 4, 7],
            [7, 4, 6],
            [4, 0, 6],
            [6, 0, 2],
            [1, 5, 3],
            [3, 5, 7],
            [2, 3, 6],
            [6, 3, 7],
            [4, 5, 0],
            [0, 5, 1],
        ]
    )
    vertex_colors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
        ]
    )

    cube2 = TriangleMesh(vertices, triangles, vertex_colors=vertex_colors)

    return [cube1, tet1, tet2, cube2]


if __name__ == "__main__":
    im_w = 800
    im_h = 600
    vertices = np.array(
        [
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5],
        ]
    )
    triangles = np.array(
        [
            [0, 1, 2],
            [2, 1, 3],
            [5, 4, 7],
            [7, 4, 6],
            [4, 0, 6],
            [6, 0, 2],
            [1, 5, 3],
            [3, 5, 7],
            [2, 3, 6],
            [6, 3, 7],
            [4, 5, 0],
            [0, 5, 1],
        ]
    )
    triangle_colors = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ]
    )
    cube = TriangleMesh(vertices, triangles, triangle_colors)

    # NOTE for your own testing purposes:
    # Uncomment and run each of these commented-out functions after you've filled them out

    # render_viewport(cube, im_w, im_h)
    ortho_vertices = np.array(
        [
            [1.0, 1.0, 1.5],
            [11.0, 1.0, 1.5],
            [1.0, 11.0, 1.5],
            [11.0, 11.0, 1.5],
            [1.0, 1.0, -1.5],
            [11.0, 1.0, -1.5],
            [1.0, 11.0, -1.5],
            [11.0, 11.0, -1.5],
        ]
    )
    ortho_cube = TriangleMesh(ortho_vertices, triangles, triangle_colors)
    # render_ortho(ortho_cube, im_w, im_h)

    # render_camera(ortho_cube, im_w, im_h)

    # render_perspective(cube, im_w, im_h)
    vertex_colors = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    cube.vertex_colors = vertex_colors
    # render_zbuffer_with_color(cube, im_w, im_h)

    objlist = get_big_scene()
    # render_big_scene(objlist, im_w, im_h)
    img = read_image("flag.png")
    uvs = my_cube_uvs(cube)
    texture_map(cube, uvs, img, im_w, im_h)
