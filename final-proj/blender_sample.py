from __future__ import annotations
import os
import numpy as np
from math import pi
import bpy
import bmesh
from mathutils import Vector, Euler
import sys
from pathlib import Path
from typing import Sequence
import types



"""
BSpline methodology from here:
    - https://github.com/Liam-Xander/Simple-BSpline-Python/blob/main/Bsplne.py

Subdivision Algo here:
    - https://github.com/prithvi1809/Mesh-Loop-Subdivision/tree/main
"""



class BSpline:
    def __init__(
        self,
        t: Sequence[float],  # knots
        c: Sequence[float],  # control points
        d: int,  # degree
    ):
        """
        t = knots
        c = bspline coefficients / control points
        d = bspline degree
        """
        self.t = t
        self.c = c
        self.d = d
        assert self.is_valid()

    def is_valid(self) -> bool:
        """Check if the B-spline configuration is valid."""
        return self.d == len(self.t) - len(self.c) - 1


    def bases(self, x: float, k: int, i: int) -> float:
        """
        Evaluate the B-spline basis function i, k at input position x.
        (Note that i, k start at 0.)
        """
        if k == 0:
            if (x >= self.t[i]) and (x < self.t[i + 1]):
                return 1.0
            else:
                return 0.0
        else:

            length1 = self.t[i + k] - self.t[i]
            length2 = self.t[i + k + 1] - self.t[i + 1]

            if length1 == 0.0:
                length1 = 1.0
            if length2 == 0.0:
                length2 = 1.0

        term1 = (x - self.t[i]) / length1 * self.bases(x=x, k=k - 1, i=i)
        term2 = (self.t[i + k + 1] - x) / length2 * self.bases(x=x, k=k - 1, i=i + 1)

        return term1 + term2

    def interp(self, x: float) -> float:
        """Evaluate the B-spline at input position x."""
        #print("running interp...")
        sum = 0.0
        for i in range(len(self.c)):
            sum += self.c[i] * self.bases(x=x, k=self.d, i=i)
        return sum

# Example calling this file from the command line:
# Replace with your path to blender and the path to this file
# On Mac, it will be something like:
# /Applications/Blender.app/Contents/MacOS/blender /Users/your-user-name/Desktop/CMSC/CMSC23700/graphics-2025/final-proj/blank.blend --background --python blender_sample.py

# And on windows, it will be something like:
# C:\Program Files\Blender Foundation\Blender\blender.exe C:\Users\your-user-name\Desktop\CMSC\CMSC23700\graphics-2025\final-proj\blank.blend --background --python blender_sample.py


# NOTE:
# If you run this code from same directory that blank.blend is in, you only need to run:
# /Applications/Blender.app/Contents/MacOS/blender blank.blend --background --python blender_sample.py


class TriangleMesh:
    def __init__(self, obj_file):
        # render mesh with/without edges (True for with edges)
        self.use_freestyle = False
        bpy.data.scenes["Scene"].render.use_freestyle = self.use_freestyle
        if self.use_freestyle:
            bpy.data.linestyles["LineStyle"].thickness = 0.8  # line thickness
        bpy.data.scenes["Scene"].use_nodes = True

        # load in the mesh object
        self.mesh = self.load_obj(obj_file)

        # initialize mesh postion, rotation and material
        self.mesh_settings(self.mesh)

    def load_obj(self, obj_file):
        bpy.ops.wm.obj_import(filepath=obj_file)
        ob = bpy.context.selected_objects[0]
        if self.use_freestyle:
            self.__mark_all_edges_freestyle(ob)
        return ob

    def __mark_all_edges_freestyle(self, mesh):
        for edge in mesh.data.edges:
            edge.use_freestyle_mark = True

    def mesh_settings(self, mesh):
        # smooth shading, comment out if you have a low poly mesh
        # or want to see the faces more clearly
        #bpy.ops.object.shade_smooth()

        # apply scale
        mesh.scale = [1, 1, 1]

        # apply rotations in radians
        mesh.rotation_euler.x = 0
        mesh.rotation_euler.y = 0
        mesh.rotation_euler.z = 0

        # shift bottom vertex to sit on zero (puts the mesh on the ground plane, you may remove if you want)
        mesh.data.update()
        bpy.context.view_layer.update()
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")
        vertices = np.array([(mesh.matrix_world @ v.co) for v in mesh.data.vertices])
        mesh.location = Vector((0, 0, mesh.location.z - min(vertices[:, 2])))

        # further translate mesh position
        mesh.location.x += 0
        mesh.location.y += 0
        mesh.location.z += 0

        # hard code color and alpha values
        r, g, b, a = 1, 0, 0, 1

        # now add material
        mat = bpy.data.materials.new(name="obj_material")
        mat.use_nodes = True

        # this wipes out any materials that come with the .obj
        #mesh.data.materials.clear()

        # the principled shader is the default shader in blender
        # and the most customizable
        #principled = mat.node_tree.nodes["Principled BSDF"]
        # here, we set just the base color
        #principled.inputs["Base Color"].default_value = (r, g, b, a)

        # you can also set other properties like roughness, metallic, etc.
        # principled.inputs["Roughness"].default_value = 0.5
        # principled.inputs["Metallic"].default_value = 0.5

        # There are also default shaders for emission, glass, etc.
        # NOTE: you cannot use the principled shader and these shaders on the same mesh
        # comment out or delete the principled definition

        # GLOWING/EMISSION MATERIAL
        # emission = mat.node_tree.nodes["Emission"]
        # emission.inputs["Color"].default_value = (r, g, b, a)
        # emission.inputs["Strength"].default_value = 1.0

        # GLASS MATERIAL
        # glass = mat.node_tree.nodes["Glass BSDF"]
        # glass.inputs["Roughness"].default_value = 0
        # glass.inputs["IOR"].default_value = 1.5

        # update material
        #mesh.data.materials.append(mat)


class Scene:
    def __init__(self):
        self.setup_render_engine()
        self.cam = self.setup_camera()

    def setup_render_engine(self):
        # CPU is faster on most laptops, but if you have a GPU, change this to GPU
        bpy.context.scene.cycles.device = "GPU"

        # we default to 10 samples at 50% resolution for faster rendering
        # feel free to increase this for higher quality, though it will take much longer per frame
        bpy.context.scene.cycles.samples = 10
        bpy.context.scene.render.resolution_percentage = 50

        # denoising is inexpensive and improves quality
        # feel free to remove if you want to see the noise
        bpy.context.scene.cycles.use_denoising = True

    def setup_camera(self):
        # camera settings
        # this camera is the only thing in the blank.blend file
        cam = bpy.data.scenes["Scene"].objects["Camera"]
        # cam.rotation_euler.x = np.deg2rad(67.5)#pi/2
        # cam.rotation_euler.y = 0
        # cam.rotation_euler.z = pi/2 
        # cam.location.x = 17
        # cam.location.y = 0
        # cam.location.z = 10

        return cam

    def add_lights(self):
        # lighting
        # feel free to adjust the light settings or add more lights
        # we will be using an area light
        lights = []
        bpy.ops.object.light_add(
            type="AREA", align="WORLD", location=(0, 0, 0), scale=(3, 3, 3)
        )
        # but you can also add other types of lights, like point lights
        # bpy.ops.object.light_add(type='POINT', radius=1, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        light = bpy.context.object
        light.data.energy = 1000
        light.data.color = (0.8, 0.77, 0.8)
        light.data.size = 10
        light.location.x = 0
        light.location.z = 7
        light.rotation_euler.y = 0.436

        lights.append(light)
        return lights

    def add_plane(self):
        # add the ground plane
        bpy.ops.mesh.primitive_plane_add(
            enter_editmode=False, align="WORLD", location=(0, 0, 0), scale=(1, 1, 1)
        )
        plane = bpy.context.object
        plane.scale.x = 100
        plane.scale.y = 100

        # this makes the plane invisible, but it will still catch shadows
        # feel free to remove if you want to see the plane, or even assign a material to it
        # plane.is_shadow_catcher = True

        # subdivide the plane (this fixes the freestyle lines)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.subdivide(number_cuts=1000)
        bpy.ops.object.mode_set(mode="OBJECT")
        plane.data.update()

        # if you want to add a material to the plane, uncomment the following lines
        # and set the shadow catcher to false
        mat = bpy.data.materials.new(name="plane_material")
        mat.use_nodes = True
        principled = mat.node_tree.nodes["Principled BSDF"]
        r, g, b, a = 0, 1, .1, .8
        principled.inputs["Base Color"].default_value = (r, g, b, a)
        plane.data.materials.append(mat)

        return plane


def select_obj(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def setup_animation_keyframes(
    object_to_animate, locations, rotations, scales, n_frames
):
    """
    insert keyframes along path
    """
    select_obj(object_to_animate)
    bpy.data.scenes["Scene"].frame_end = int(n_frames - 1)
    # f = 1

    # for loc, rot, scale in zip(locations, rotations, scales):
    #     print(loc, rot, scale)
    #     # put object at location
    #     object_to_animate.location[0] = loc[0]
    #     object_to_animate.location[1] = loc[1]
    #     object_to_animate.location[2] = loc[2]

    #     # rotations
    #     object_to_animate.rotation_euler[0] = rot[0]
    #     object_to_animate.rotation_euler[1] = rot[1]
    #     object_to_animate.rotation_euler[2] = rot[2]

    #     # scale
    #     object_to_animate.scale[0] = scale[0]
    #     object_to_animate.scale[1] = scale[1]
    #     object_to_animate.scale[2] = scale[2]

    #     bpy.data.scenes["Scene"].frame_current = f
    #     object_to_animate.keyframe_insert(data_path="location", frame=f)
    #     object_to_animate.keyframe_insert(data_path="rotation_euler", frame=f)
    #     object_to_animate.keyframe_insert(data_path="scale", frame=f)
    #     f += 1

def insert_rest_keyframes(object, frame_count):
    anim = Animation(object, frame_count)
    insert_animation_keyframes(anim)

def insert_animation_keyframes(animation:Animation):
    object_to_animate = animation.object
    locations = animation.get_location_frames()
    rotations = animation.get_rotation_frames()
    scales = animation.get_scale_frames()
    select_obj(object_to_animate)
    if  bpy.data.scenes["Scene"].frame_current == 1:
        print("setting frame 1")
        f = 1
    else:
        f = bpy.data.scenes["Scene"].frame_current+1

    if animation.frame_count == 0:
        raise Exception("cannot have an animation with no frames!")


    for loc, rot, scale in zip(locations, rotations, scales):
        print(f"Inserting {loc}, {rot}, {scale}")

        object_to_animate.location[0] = loc[0]
        object_to_animate.location[1] = loc[1]
        object_to_animate.location[2] = loc[2]

        # rotations
        object_to_animate.rotation_euler[0] = rot[0]
        object_to_animate.rotation_euler[1] = rot[1]
        object_to_animate.rotation_euler[2] = rot[2]

        # scale
        object_to_animate.scale[0] = scale[0]
        object_to_animate.scale[1] = scale[1]
        object_to_animate.scale[2] = scale[2]

        bpy.data.scenes["Scene"].frame_current = f
        object_to_animate.keyframe_insert(data_path="location", frame=f)
        object_to_animate.keyframe_insert(data_path="rotation_euler", frame=f)
        object_to_animate.keyframe_insert(data_path="scale", frame=f)
        f += 1


# ---------- Scene Setup ---------- #
# set scene parameters and camera

scene = Scene()
cam = scene.cam

def midpoint(v1:float, v2:float):
    x = (v1[0] + v2[0]) / 2
    y = (v1[1] + v2[1]) / 2
    z = (v1[2] + v2[2]) / 2
    return (x,y,z)

def loop_subdivision(mesh):
    bm = bmesh.new()
    bm.from_mesh(mesh.data)
    new_vertices = {}
    new_faces = []
    next_index = len(bm.verts)
    for f in bm.faces:
        v1, v2, v3 = [v for v in f.verts]



        # Get the midpoint between each vertex pair
        m1 = midpoint(v1.co, v2.co)
        m2 = midpoint(v2.co, v3.co)
        m3 = midpoint(v3.co, v1.co)

        for v in [m1, m2, m3]:
            if v not in list(new_vertices.keys()):
                # new vertex!
                new_vertices[v] = next_index
                next_index += 1
            else:
                # duplicate vertex
                pass

        new_faces.append((v1.index, new_vertices[m1], new_vertices[m3]))
        new_faces.append((v2.index, new_vertices[m1], new_vertices[m2]))
        new_faces.append((v3.index, new_vertices[m2], new_vertices[m3]))
        new_faces.append((new_vertices[m1], new_vertices[m2], new_vertices[m3]))

        #print(f"face {f.index} finished. created face with vertices at {new_vertices[m1]}, {new_vertices[m2]} and {new_vertices[m3]}")
    # update old vertex poistions based on neighbors
    updated_verts = update_old_verts(bm)
    print("updated old verts")

    # put the object in edit mode so we can fiddle with it's vertices
    obj = bpy.context.selected_objects[0]
    bpy.ops.object.mode_set(mode="EDIT")
    subdivided_mesh = bmesh.from_edit_mesh(obj.data)

    # replace all the existing vertex locations with their updated ones
    for v in subdivided_mesh.verts:
        v.co = updated_verts[v.index]

    # add all the newly created vertices
    for v in new_vertices.keys():
        subdivided_mesh.verts.new(v)

    # remove all the olds faces
    for f in subdivided_mesh.faces:
        subdivided_mesh.faces.remove(f)

    # fix the vertex table
    subdivided_mesh.verts.ensure_lookup_table()
    subdivided_mesh.verts.index_update()

    # add all the new faces created by subdividing
    for face in new_faces:
        subdivided_mesh.faces.new((
            subdivided_mesh.verts[face[0]],
            subdivided_mesh.verts[face[1]],
            subdivided_mesh.verts[face[2]]
        ))

    # apply the edit mode edits to the object and exit edit mode
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')
    #bpy.ops.wm.obj_export(filepath=subdivided, export_selected_objects=True)

def update_old_verts(mesh_obj):
    new = []

    for v in mesh_obj.verts:
        #print(f"getting neighbors of vertex {v.index}")
        neighbors = get_vertex_neighbors(v)
        n = len(neighbors)

        if n == 0:
            raise Exception("vertex has no neighbors!")
        else:
            if n == 3:
                beta = 3 / 16
            elif n < 3:
                beta = 3 / (8*n)
            else:
                beta = 1/n * (5/8 - (3/8 + 0.5 * np.cos(2 * np.pi / n))**2)

            new_pos = (1.0 - n * beta) * v.co
            for neighbor in neighbors:
                new_pos += beta * neighbor.co

        new.append(new_pos)

    return new

def get_vertex_neighbors(vertex):
    return [edge.other_vert(vertex) for edge in vertex.link_edges]

# ---------- Load objs ---------- #
# set obj file to spot.obj in the meshes directory
# we've also included some other .obj files, but feel free to download or create your own
pikachu_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "meshes", "Pikachu.obj"
)

pokeball_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meshes", "pokeball.obj"
                             
)

platform_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meshes", "platform.obj"
                             
)

print(f"pikachu file: {pikachu_file}")
print(f"pokeball file: {pokeball_file}")

pikachu = TriangleMesh(pikachu_file)
pikachu = pikachu.mesh


# load in obj file and get mesh object
pokeball = TriangleMesh(pokeball_file)
pokeball = pokeball.mesh

platform = TriangleMesh(platform_file)
platform = platform.mesh

subdivided = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meshes", "subdivided.obj")
loop_subdivision(platform)
loop_subdivision(platform)
loop_subdivision(platform)

# Initial setup, i.e. move objects to their starting position
pokeball.rotation_euler[0] = np.deg2rad(90)
pikachu.rotation_euler[0] = np.deg2rad(90)
platform.rotation_euler[0] = np.deg2rad(90)

pokeball.scale = (0.5, 0.5, 0.5)
pikachu.location[0] = -5.0
pikachu.location[2] += 1.2
pokeball.location[0] = 4.5
pokeball.location[2] += 0.5
platform.location[0] = -5.0
platform.location[2] -= 2.4



# ---------- Render Mesh ---------- #
# setup output file path
output_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output", "renders", "spot.png"
)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

print(f"output path: {output_path}")
# edit output path as needed
bpy.data.scenes["Scene"].render.filepath = output_path

# debug flag
debug = False


# ---------- Animation ---------- #
# animate flag
animate = True

# hard code locations
n_frames = 45

# example animations (moving either cam or mesh)
# example camera animation
#locations = zip([6.6]*n_frames, np.linspace(-6, 3, n_frames), [2.49]*n_frames)
#rotations = zip([1.36]*n_frames, [0]*n_frames, [1.157]*n_frames)

# example mesh animation
# this uses np.linspace to generate linear animations
# but you can replace this with your splines!

def generate_bspline_curve(controls, frames):
    degree = len(controls) - 1
    min_val = min(controls)
    max_val = max(controls)
    knots = np.concatenate((
        [min_val] * degree,
        np.linspace(min_val, max_val, (len(controls) - degree + 1)),
        [max_val] * degree,
    ))

    spline = BSpline(knots, controls, degree)
    x = np.linspace(min_val, max_val, frames)
    #print(f"Interpolating linspace: {x}\n")
    inter = [spline.interp(pos) for pos in x]
    #print(f"Interpolated the following results: {inter}\n")
    return inter

def get_rotations(min_degree, max_degree, frames=n_frames):
    if min_degree < 0:
        min = np.deg2rad(min_degree)
        max = np.deg2rad(max_degree)
        controls = [min/2, min, min/2, (min+max)/2, max/2, max, max/2, (min+max)/2]
        return generate_bspline_curve(controls, frames)

def get_locations(start, end, frames=n_frames):
    if start > end:
        # going down
        controls = [start, (start+end)*0.75, (start+end)*0.25, end]
    else:
        # going up
        controls = [start, (start+end)*0.25, (start+end)*0.75, end]
    return generate_bspline_curve(controls, frames)[:-1]+[end]

def get_rotation_by(start, degree, frames=n_frames):
    """
    Generate roation animation frames. Frames start at 'start'
    and are rotated by 'degree' degrees.
    """
    end = np.deg2rad(degree) + start
    controls = [start, end]

    #print(f"getting rotation from {start} to {end}")

    inter = generate_bspline_curve(controls, frames)
    inter[-1] = end
    return inter

def get_rotation_to(start, end, frames=n_frames):
    """
    Generate frames for a rotation beginning at start and ending at end
    """
    return get_locations(np.deg2rad(start), np.deg2rad(end), frames) 

#defaults
class Animation:

    def __init__(self, object, frame_count):
        self.object = object
        self.frame_count = frame_count
        self._location_frames:Sequence | None = None
        self._rotation_frames = None
        self._scale_frames = None

    def get_location_frames(self):
        return zip(
            [self.object.location[0]] * self.frame_count, 
            [self.object.location[1]] * self.frame_count, 
            [self.object.location[2]] * self.frame_count
        ) if not self._location_frames else self._location_frames
    
    def get_rotation_frames(self):
        return zip(
            [self.object.rotation_euler[0]] * self.frame_count, 
            [self.object.rotation_euler[1]] * self.frame_count, 
            [self.object.rotation_euler[2]] * self.frame_count
        ) if not self._rotation_frames else self._rotation_frames

    def get_scale_frames(self):
        return zip(
            [self.object.scale[0]] * self.frame_count, 
            [self.object.scale[1]] * self.frame_count, 
            [self.object.scale[2]] * self.frame_count
        ) if not self._scale_frames else self._scale_frames


fall = Animation(object=pokeball, frame_count=25)
fall._location_frames = zip(
    [fall.object.location[0]] * fall.frame_count,
    [fall.object.location[1]] * fall.frame_count,
    get_locations(10, fall.object.location[2], fall.frame_count)[:-1]+[fall.object.location[2]]
)


big_wiggle = Animation(pokeball, 30)
big_wiggle._rotation_frames = zip(
    [big_wiggle.object.rotation_euler[0]] * big_wiggle.frame_count, 
    get_rotations(-65, 65, big_wiggle.frame_count), 
    [big_wiggle.object.rotation_euler[2]] * big_wiggle.frame_count
)

med_wiggle = Animation(pokeball, 30)
med_wiggle._rotation_frames = zip(
    [med_wiggle.object.rotation_euler[0]] * med_wiggle.frame_count, 
    get_rotations(-45, 45, med_wiggle.frame_count), 
    [med_wiggle.object.rotation_euler[2]] * med_wiggle.frame_count
)

small_wiggle = Animation(pokeball, 30)
small_wiggle._rotation_frames = zip(
    [small_wiggle.object.rotation_euler[0]] * small_wiggle.frame_count, 
    get_rotations(-25, 25, small_wiggle.frame_count), 
    [small_wiggle.object.rotation_euler[2]] * small_wiggle.frame_count
)

pokeball_left = Animation(pokeball, 30)
pokeball_left._rotation_frames = zip(
    [pokeball.rotation_euler[0]] * pokeball_left.frame_count, 
    [pokeball.rotation_euler[1]] * pokeball_left.frame_count, 
    get_rotation_by(pokeball.rotation_euler[2], -90, pokeball_left.frame_count),
)

pokeball_return_to_center = Animation(pokeball, 30)
pokeball_return_to_center._rotation_frames = zip(
    [pokeball.rotation_euler[0]] * pokeball_return_to_center.frame_count, 
    [pokeball.rotation_euler[1]] * pokeball_return_to_center.frame_count, 
    get_rotation_to(-90, 0, pokeball_return_to_center.frame_count),
)

pikachu_right = Animation(pikachu, 30)
pikachu_right._rotation_frames = zip(
    [pikachu.rotation_euler[0]] * pikachu_right.frame_count, 
    [pikachu.rotation_euler[1]] * pikachu_right.frame_count, 
    get_rotation_by(pikachu_right.object.rotation_euler[2], 90, pikachu_right.frame_count)
)

pikachu_captured = Animation(pikachu, 30)
pikachu_captured._scale_frames = zip(
    get_locations(pikachu_captured.object.scale[0], 0.001, pikachu_captured.frame_count),
    get_locations(pikachu_captured.object.scale[1], 0.001, pikachu_captured.frame_count),
    get_locations(pikachu_captured.object.scale[2], 0.001, pikachu_captured.frame_count),
)

pikachu_captured._location_frames = zip (
    get_locations(pikachu_captured.object.location[0], pokeball.location[0]+1, pikachu_captured.frame_count),
    [pikachu_captured.object.location[1]] * pikachu_captured.frame_count,
    get_locations(pikachu_captured.object.location[2], pokeball.location[2], pikachu_captured.frame_count)
)

zoom = Animation(cam, 30)
zoom._location_frames = zip(
    get_locations(zoom.object.location[0], pokeball.location[0]+0.5, zoom.frame_count),
    get_locations(zoom.object.location[1], pokeball.location[1]-3, zoom.frame_count),
    get_locations(zoom.object.location[2], pokeball.location[2]+0.75, zoom.frame_count)
)

#    get_locations(zoom.object.location[1], zoom.object.location[1]+5, zoom.frame_count),
#    get_locations(zoom.object.location[2], zoom.object.location[2]-0.3, zoom.frame_count)

if animate:
#     # setup_animation_keyframes(cam, locations, rotations, scales, n_frames) # for camera
#     # setup_animation_keyframes(pokeball, default_loc, default_rot, default_scale, n_frames)  # setup animation frames
    bpy.data.scenes["Scene"].frame_end = int(n_frames)
    #insert_animation_keyframes(fall) # 30 frames == 1sec

    #insert_animation_keyframes(pokeball_left) # 30 frames = 1sec

    #pokeball.rotation_euler[2] = np.deg2rad(-90)
    pikachu.rotation_euler[2] = np.deg2rad(90)

    pikachu.scale = (0.001, 0.001, 0.001)
    pikachu.location = (5.5, 0.0, 1.5284843444824219)

    cam.location = (4.574705809996311, -6.652732789372261, 2.2585730261163146)

    cam.rotation_euler = (1.4311699867248535, 0.0, 0.0)

    #insert_animation_keyframes(pikachu_right) # 30 frames = 1sec

    #insert_animation_keyframes(pikachu_captured) # 30 frames == 1sec

    #insert_animation_keyframes(pokeball_return_to_center) # 30 frames = 1sec

    #insert_animation_keyframes(zoom) # 30 frames == 1sec

    #insert_rest_keyframes(pokeball, 3)

    #insert_animation_keyframes(big_wiggle) # 30 frames == 1sec

    #insert_rest_keyframes(pokeball, 15) # ==> .5 sec

    #insert_animation_keyframes(med_wiggle) # ==> 30 frames == 1sec

    insert_rest_keyframes(pokeball, 10) # ==> .5 sec

    insert_animation_keyframes(small_wiggle) #==> 30 frames == 1sec

    insert_rest_keyframes(pokeball, 5)


    Path(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "output", "animation_renders"
        )
    ).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "output",
        "animation_renders",
        "spot.png",
    )  # edit output path as needed
    bpy.data.scenes["Scene"].render.filepath = output_path

# render
if not debug:
    bpy.ops.render.render(write_still=not animate, animation=animate)
else:
    print("debugging...")
    bpy.ops.wm.save_as_mainfile(filepath="/tmp/debug.blend")
    # To debug, run the following code. This will open up the debug file in blender.
    # path/to/blender /tmp/debug.blend
    # For me this looks like:
    # /Applications/Blender.app/Contents/MacOS/blender /tmp/debug.blend
