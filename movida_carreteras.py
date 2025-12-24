bl_info = {
    "name": "Road Builder (Mesh, Cities-Style) + Vertex Colors + Solid Road",
    "author": "Vlaskovich",
    "version": (1, 1, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > Road Builder",
    "description": "Draw roads by extruding selected edge on a mesh tile, with optional curbs/markings, vertex color painting, and solid road volume.",
    "category": "Mesh",
}

import bpy
import bmesh
from mathutils import Vector
from bpy.props import (
    BoolProperty, FloatProperty, PointerProperty, StringProperty
)
from bpy.types import Operator, Panel, PropertyGroup
from bpy_extras import view3d_utils

UP = Vector((0.0, 0.0, 1.0))


# -----------------------------
# Helpers
# -----------------------------
def ensure_edit_mesh(context):
    obj = context.object
    if obj is None or obj.type != 'MESH':
        return None
    if context.mode != 'EDIT_MESH':
        return None
    return obj


def get_selected_edge(bm):
    # Prefer active edge, else first selected
    active = bm.select_history.active
    if active and isinstance(active, bmesh.types.BMEdge) and active.select:
        return active
    for ed in bm.edges:
        if ed.select:
            return ed
    return None


def raycast_mouse_to_scene(context, event, max_dist=10000.0):
    """Raycast from mouse into scene, return (hit, location, normal)."""
    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None:
        return (False, None, None)

    coord = (event.mouse_region_x, event.mouse_region_y)
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord).normalized()

    depsgraph = context.evaluated_depsgraph_get()
    hit, loc, normal, _, _, _ = context.scene.ray_cast(depsgraph, origin, direction, distance=max_dist)
    return (hit, loc, normal)


def raycast_down(context, point_w, max_dist=10000.0):
    """Raycast straight down from above a WORLD point. Returns (hit, loc_w, normal_w)."""
    depsgraph = context.evaluated_depsgraph_get()
    origin = point_w + UP * 50.0
    direction = -UP
    hit, loc, normal, _, _, _ = context.scene.ray_cast(depsgraph, origin, direction, distance=max_dist)
    return hit, loc, normal


def clamp(x, a, b):
    return max(a, min(b, x))


def set_only_edge_selected(bm, edge):
    for e in bm.edges:
        e.select = False
    for v in bm.verts:
        v.select = False
    edge.select = True
    edge.verts[0].select = True
    edge.verts[1].select = True
    bm.select_history.clear()
    bm.select_history.add(edge)


# -----------------------------
# Vertex Color helpers (BMesh loops color layer)
# -----------------------------
def _normalize_hex(h: str) -> str:
    h = (h or "").strip()
    if not h:
        return "#000000"
    if not h.startswith("#"):
        h = "#" + h
    if len(h) == 4:  # #RGB -> #RRGGBB
        h = "#" + "".join([c * 2 for c in h[1:]])
    if len(h) != 7:
        return "#000000"
    return h


def hex_to_rgba(hex_color: str, alpha: float = 1.0):
    hex_color = _normalize_hex(hex_color).lstrip("#")
    try:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
    except Exception:
        r, g, b = 0.0, 0.0, 0.0
    return (r, g, b, float(alpha))


def ensure_bmesh_color_layer(bm: bmesh.types.BMesh, name: str):
    """
    Ensures a loop color layer exists in BMesh.
    Blender will back this by a Color Attribute (domain CORNER).
    """
    layer = bm.loops.layers.color.get(name)
    if layer is None:
        layer = bm.loops.layers.color.new(name)
    return layer


def paint_face(face: bmesh.types.BMFace, color_layer, rgba):
    if face is None:
        return
    for loop in face.loops:
        loop[color_layer] = rgba


# -----------------------------
# Solidify (geometry, no modifier)
# -----------------------------
def solidify_face_prism(bm: bmesh.types.BMesh, face: bmesh.types.BMFace, depth: float):
    """
    Create a closed prism by duplicating face vertices downward along -Z (local),
    creating bottom face + side faces. Returns list of created faces (including bottom/sides).
    """
    created_faces = []
    if face is None:
        return created_faces

    depth = max(0.0, float(depth))
    if depth <= 1e-6:
        return created_faces

    top_verts = list(face.verts)
    bottom_verts = []

    for v in top_verts:
        vb = bm.verts.new(v.co - Vector((0, 0, depth)))
        bottom_verts.append(vb)

    # Bottom face (reverse order to keep normals outward-ish)
    try:
        f_bottom = bm.faces.new(tuple(reversed(bottom_verts)))
        created_faces.append(f_bottom)
    except ValueError:
        f_bottom = None

    # Side faces
    n = len(top_verts)
    for i in range(n):
        v0 = top_verts[i]
        v1 = top_verts[(i + 1) % n]
        v2 = bottom_verts[(i + 1) % n]
        v3 = bottom_verts[i]
        try:
            f_side = bm.faces.new((v0, v1, v2, v3))
            created_faces.append(f_side)
        except ValueError:
            pass

    return created_faces


# -----------------------------
# Settings
# -----------------------------
class RB_Settings(PropertyGroup):
    road_width: FloatProperty(
        name="Road Width",
        default=6.0,
        min=0.2,
        soft_max=50.0,
        subtype='DISTANCE'
    )

    min_segment: FloatProperty(
        name="Min Segment",
        default=1.0,
        min=0.1,
        soft_max=50.0,
        subtype='DISTANCE'
    )

    max_segment: FloatProperty(
        name="Max Segment",
        default=12.0,
        min=0.2,
        soft_max=200.0,
        subtype='DISTANCE'
    )

    snap_to_surface: BoolProperty(
        name="Snap to Surface",
        default=True,
        description="Raycast to place the road on top of the tile surface"
    )

    conform_each_vertex: BoolProperty(
        name="Conform Each Vertex",
        default=False,
        description="Raycast down for each new vertex (more accurate, a bit heavier)"
    )

    z_offset: FloatProperty(
        name="Z Offset",
        default=0.01,
        min=-1.0,
        max=1.0,
        description="Small offset to avoid z-fighting with the tile"
    )

    # Curbs
    add_curbs: BoolProperty(
        name="Add Curbs",
        default=False
    )

    curb_out: FloatProperty(
        name="Curb Out",
        default=0.35,
        min=0.0,
        soft_max=5.0,
        subtype='DISTANCE',
        description="How far curbs extend outward"
    )

    curb_up: FloatProperty(
        name="Curb Up",
        default=0.12,
        min=0.0,
        soft_max=2.0,
        subtype='DISTANCE',
        description="How high curbs are raised"
    )

    # Markings
    add_markings: BoolProperty(
        name="Add Markings",
        default=False
    )

    marking_width: FloatProperty(
        name="Marking Width",
        default=0.15,
        min=0.01,
        soft_max=2.0,
        subtype='DISTANCE'
    )

    marking_inset: FloatProperty(
        name="Marking Inset",
        default=0.15,
        min=0.0,
        soft_max=2.0,
        subtype='DISTANCE',
        description="Inset from segment ends to avoid overlapping markings at joins"
    )

    # Vertex Colors
    color_layer_name: StringProperty(
        name="Color Layer",
        default="RB_Color",
        description="Name of the vertex color / color attribute layer (CORNER/loop)"
    )

    road_hex: StringProperty(
        name="Road HEX",
        default="#2a2a2a",
        description="HEX color for road faces (e.g. #2a2a2a)"
    )

    marking_hex: StringProperty(
        name="Marking HEX",
        default="#ffffff",
        description="HEX color for markings faces (e.g. #ffffff)"
    )

    curb_hex: StringProperty(
        name="Curb HEX",
        default="#777777",
        description="HEX color for curb faces"
    )

    # Solid road volume
    solidify_road: BoolProperty(
        name="Solid Road",
        default=False,
        description="Give the road volume by extruding downward (no modifiers)"
    )

    solid_depth: FloatProperty(
        name="Solid Depth",
        default=0.30,
        min=0.01,
        soft_max=10.0,
        subtype='DISTANCE',
        description="Depth of the solid road volume"
    )

    use_knife_flatten: bpy.props.BoolProperty(
        name="Knife Flatten",
        default=False,
        description="Cut terrain using Knife Project with a temporary cutter per segment (best for low poly)."
    )

    knife_cut_through: bpy.props.BoolProperty(
        name="Cut Through",
        default=False,
        description="Knife Project cut through the whole mesh. Usually OFF for terrain."
    )

    knife_margin: bpy.props.FloatProperty(
        name="Knife Margin",
        default=0.0,
        min=0.0,
        soft_max=2.0,
        subtype='DISTANCE',
        description="Extra width added to the cutter (expands road footprint)."
    )

    knife_flatten_strength: bpy.props.FloatProperty(
        name="Knife Strength",
        default=1.0,
        min=0.0,
        soft_max=5.0,
        description="Flatten intensity after cutting (can be >1 for low poly)."
    )

    flatten_terrain: bpy.props.BoolProperty(
        name="Flatten Terrain",
        default=False,
        description="Deform terrain vertices to flatten under the road"
    )

    flatten_radius: bpy.props.FloatProperty(
        name="Flatten Radius",
        default=6.0,
        min=0.1,
        soft_max=50.0,
        subtype='DISTANCE',
        description="Radius of terrain deformation around the road"
    )

    flatten_strength: bpy.props.FloatProperty(
        name="Flatten Strength",
        default=1.0,
        min=0.0,
        max=10.0,
        description="How strongly terrain is flattened (0-1)"
    )

    remove_inner_terrain: bpy.props.BoolProperty(
        name="Remove Inner Terrain",
        default=True,
        description="Delete terrain faces inside the road footprint after Knife Project."
    )

    stitch_terrain_to_road: bpy.props.BoolProperty(
        name="Stitch Terrain to Road",
        default=True,
        description="Snap nearby terrain verts to the road border and weld."
    )

    stitch_distance: bpy.props.FloatProperty(
        name="Stitch Distance",
        default=0.25,
        min=0.0,
        soft_max=2.0,
        subtype='DISTANCE',
        description="How far terrain vertices can be from the road border to be stitched."
    )

    merge_road_border: bpy.props.BoolProperty(
        name="Merge Road Border",
        default=True,
        description="Merge terrain border verts into the road border so they share vertices."
    )

    merge_dist: bpy.props.FloatProperty(
        name="Merge Distance",
        default=0.02,
        min=0.0,
        soft_max=10,
        subtype='DISTANCE',
        description="Distance used to merge road border with terrain verts."
    )

def ensure_face_kind_layer(bm, name="RB_KIND"):
    layer = bm.faces.layers.int.get(name)
    if layer is None:
        layer = bm.faces.layers.int.new(name)
    return layer

def set_face_kind(face, kind_layer, kind: int):
    if face and kind_layer:
        face[kind_layer] = int(kind)

def get_face_kind(face, kind_layer, default=0):
    try:
        return int(face[kind_layer])
    except Exception:
        return default


def snap_and_merge_terrain_to_road_border(
    bm: bmesh.types.BMesh,
    a0_co: Vector, a1_co: Vector,
    b0_co: Vector, b1_co: Vector,
    stitch_dist: float,
    merge_dist: float,
    target_z: float = None
):
    """
    - Encuentra verts del terreno cerca de los bordes de carretera (izq y der).
    - Los proyecta al borde (snap XY), opcional Z.
    - Hace remove_doubles incluyendo también los verts de carretera del borde.
    Devuelve las coords (pueden usarse para remapeo posterior).
    """
    if stitch_dist <= 0.0:
        return

    moved = []
    road_candidates = []

    # remapeo local: busca los 4 verts de borde en el bm actual
    a0v = find_vert_near(bm, a0_co)
    a1v = find_vert_near(bm, a1_co)
    b0v = find_vert_near(bm, b0_co)
    b1v = find_vert_near(bm, b1_co)
    if None in (a0v, a1v, b0v, b1v):
        return

    # coords (por si se cambian ligeramente)
    aL = a0v.co.copy(); bL = a1v.co.copy()
    aR = b0v.co.copy(); bR = b1v.co.copy()

    road_candidates.extend([a0v, a1v, b0v, b1v])

    for v in bm.verts:
        # no mover los de carretera
        if v in (a0v, a1v, b0v, b1v):
            continue

        dL, cpL = distance_point_to_segment_xy(v.co, aL, bL)
        dR, cpR = distance_point_to_segment_xy(v.co, aR, bR)

        if dL <= stitch_dist or dR <= stitch_dist:
            cp = cpL if dL <= dR else cpR
            v.co.x = cp.x
            v.co.y = cp.y
            if target_z is not None:
                v.co.z = target_z
            moved.append(v)

    # Merge real: incluye también el borde de carretera
    if moved and merge_dist > 0.0:
        verts_to_merge = moved + road_candidates
        bmesh.ops.remove_doubles(bm, verts=verts_to_merge, dist=merge_dist)


def find_vert_near(bm: bmesh.types.BMesh, co: Vector, eps=1e-4):
    best = None
    best_d2 = 1e18
    for v in bm.verts:
        d2 = (v.co - co).length_squared
        if d2 < best_d2:
            best_d2 = d2
            best = v
    if best is None:
        return None
    # si quieres, puedes exigir eps:
    # if best_d2 > eps*eps: return None
    return best


def closest_point_on_segment_xy(p: Vector, a: Vector, b: Vector):
    """Closest point to p on segment a-b, considering XY only (Z ignored). Returns (point, t)."""
    ax, ay = a.x, a.y
    bx, by = b.x, b.y
    px, py = p.x, p.y
    abx = bx - ax
    aby = by - ay
    denom = abx*abx + aby*aby
    if denom < 1e-12:
        return Vector((ax, ay, p.z)), 0.0
    t = ((px-ax)*abx + (py-ay)*aby) / denom
    t = max(0.0, min(1.0, t))
    cx = ax + abx*t
    cy = ay + aby*t
    return Vector((cx, cy, p.z)), t


def distance_point_to_segment_xy(p: Vector, a: Vector, b: Vector):
    cp, _ = closest_point_on_segment_xy(p, a, b)
    dx = p.x - cp.x
    dy = p.y - cp.y
    return (dx*dx + dy*dy) ** 0.5, cp


def point_in_road_capsule_xy(p: Vector, c0: Vector, c1: Vector, radius: float):
    """Is p within capsule around segment c0-c1 in XY."""
    d, _ = distance_point_to_segment_xy(p, c0, c1)
    return d <= radius


def delete_faces_inside_capsule(bm, c0, c1, radius, kind_layer=None, terrain_kind=0):
    to_del = []
    for f in bm.faces:
        if kind_layer and get_face_kind(f, kind_layer, 0) != terrain_kind:
            continue  # no es terreno, no se toca

        all_inside = True
        for v in f.verts:
            if not point_in_road_capsule_xy(v.co, c0, c1, radius):
                all_inside = False
                break
        if all_inside:
            to_del.append(f)

    if to_del:
        bmesh.ops.delete(bm, geom=to_del, context='FACES')



def stitch_verts_to_road_edges_xy(
    bm: bmesh.types.BMesh,
    aL: Vector, bL: Vector,
    aR: Vector, bR: Vector,
    max_dist: float,
    weld_dist: float = None,
    target_z: float = None,
    protect_eps: float = 1e-6
):
    """
    Snap verts near the left/right road border segments onto those segments in XY,
    optionally set Z, and weld close verts.
    NO depende de BMVerts vivos (solo coords).
    """
    if max_dist <= 0.0:
        return

    moved = []

    # helper: no tocar los puntos exactos de los bordes (evita mover la carretera)
    def is_protected(vco: Vector) -> bool:
        return ((vco - aL).length_squared < protect_eps or
                (vco - bL).length_squared < protect_eps or
                (vco - aR).length_squared < protect_eps or
                (vco - bR).length_squared < protect_eps)

    for v in bm.verts:
        if is_protected(v.co):
            continue

        dL, cpL = distance_point_to_segment_xy(v.co, aL, bL)
        dR, cpR = distance_point_to_segment_xy(v.co, aR, bR)

        if dL <= max_dist or dR <= max_dist:
            cp = cpL if dL <= dR else cpR

            v.co.x = cp.x
            v.co.y = cp.y

            if target_z is not None:
                v.co.z = target_z

            moved.append(v)

    if moved:
        if weld_dist is None:
            weld_dist = max_dist * 0.5
        bmesh.ops.remove_doubles(bm, verts=moved, dist=weld_dist)




def flatten_terrain_around_segment(
    bm: bmesh.types.BMesh,
    road_mid: Vector,
    road_z: float,
    radius: float,
    strength: float,
    hard_core_ratio: float = 0.3  
):
    if radius <= 0.0 or strength <= 0.0:
        return

    r2 = radius * radius
    hard_r = max(0.0, min(1.0, hard_core_ratio)) * radius
    hard_r2 = hard_r * hard_r

    for v in bm.verts:
        dx = v.co.x - road_mid.x
        dy = v.co.y - road_mid.y
        dist2 = dx*dx + dy*dy
        if dist2 > r2:
            continue

        # Zona dura: pegado total (ideal low poly)
        if hard_r > 0.0 and dist2 <= hard_r2:
            v.co.z = road_z
            continue

        dist = dist2 ** 0.5
        t = 1.0 - (dist / radius)
        if t <= 0.0:
            continue

        # Falloff suave (puedes cambiar a t**3 si quieres más borde definido)
        t = t * t

        # Strength > 1: más agresivo, pero sin overshoot
        w = t * strength
        if w > 1.0:
            w = 1.0

        v.co.z = v.co.z * (1.0 - w) + road_z * w



def create_segment_cutter_object(context, obj, a0, b0, a1, b1, margin=0.0, z_raise=0.05):
    """
    Crea un objeto cutter (mesh) con un quad footprint del tramo, en WORLD coords.
    NO se oculta (para que sea seleccionable por Knife Project), pero se deja como wire.
    """
    mw = obj.matrix_world

    width_axis = (b0 - a0).normalized()
    a0e = a0 - width_axis * margin
    b0e = b0 + width_axis * margin
    a1e = a1 - width_axis * margin
    b1e = b1 + width_axis * margin

    pts_w = [
        mw @ (a0e + Vector((0, 0, z_raise))),
        mw @ (b0e + Vector((0, 0, z_raise))),
        mw @ (b1e + Vector((0, 0, z_raise))),
        mw @ (a1e + Vector((0, 0, z_raise))),
    ]

    mesh = bpy.data.meshes.new("RB_CutterMesh")
    cutter = bpy.data.objects.new("RB_Cutter", mesh)
    context.collection.objects.link(cutter)

    edges = [(0,1), (1,2), (2,3), (3,0)]
    mesh.from_pydata(pts_w, edges, [])
    mesh.update(calc_edges=True)


    # Para que no moleste visualmente, pero siga seleccionable:
    cutter.display_type = 'WIRE'
    cutter.show_in_front = True
    cutter.hide_render = True
    cutter.hide_select = False
    cutter.hide_viewport = False

    return cutter


def knife_project_on_object(context, target_obj, cutter_obj, cut_through=False):
    """
    Ejecuta Knife Project de forma estable:
    - Busca un área VIEW_3D
    - Fuerza vista TOP ortográfica
    - Selecciona cutter + target, y target activo en Edit Mode
    """
    prev_active = context.view_layer.objects.active
    prev_sel = [o for o in context.selected_objects]

    # Busca un VIEW_3D para override
    area = None
    region = None
    space = None
    for a in context.window.screen.areas:
        if a.type == 'VIEW_3D':
            area = a
            for r in a.regions:
                if r.type == 'WINDOW':
                    region = r
                    break
            space = a.spaces.active
            break

    if area is None or region is None or space is None or space.region_3d is None:
        raise RuntimeError("No VIEW_3D area found for Knife Project override")

    try:
        # Selección de objetos: cutter y target
        for o in context.selected_objects:
            o.select_set(False)

        cutter_obj.hide_viewport = False
        cutter_obj.hide_select = False
        cutter_obj.select_set(True)

        target_obj.select_set(True)
        context.view_layer.objects.active = target_obj

        # Asegura edit mode en target
        if context.mode != 'EDIT_MESH':
            bpy.ops.object.mode_set(mode='EDIT')

        # Override para que knife_project use una vista consistente
        override = {
            "window": context.window,
            "screen": context.window.screen,
            "area": area,
            "region": region,
            "space_data": space,
            "region_data": space.region_3d,
            "active_object": target_obj,
            "object": target_obj,
            "edit_object": target_obj,
        }

        # Fuerza TOP ortho antes del corte
        with context.temp_override(**override):
            bpy.ops.view3d.view_axis(type='TOP')
            bpy.ops.view3d.view_persportho()
            bpy.ops.mesh.knife_project(cut_through=cut_through)

    finally:
        # Restaurar selección
        if prev_active:
            context.view_layer.objects.active = prev_active
        for o in context.selected_objects:
            o.select_set(False)
        for o in prev_sel:
            o.select_set(True)



def flatten_by_segment_capsule(bm, p0, p1, target_z, radius, strength):
    """
    Aplana vértices cuya proyección cae cerca del segmento p0->p1 (en XY), tipo cápsula.
    p0,p1 coords locales.
    """
    if radius <= 0.0 or strength <= 0.0:
        return

    seg = Vector((p1.x - p0.x, p1.y - p0.y, 0.0))
    seg_len2 = seg.x*seg.x + seg.y*seg.y
    if seg_len2 < 1e-9:
        return

    for v in bm.verts:
        # trabajar en XY
        pv = Vector((v.co.x - p0.x, v.co.y - p0.y, 0.0))
        t = (pv.x*seg.x + pv.y*seg.y) / seg_len2
        t = max(0.0, min(1.0, t))

        closest = Vector((p0.x + seg.x*t, p0.y + seg.y*t, 0.0))
        dx = v.co.x - closest.x
        dy = v.co.y - closest.y
        dist = (dx*dx + dy*dy) ** 0.5

        if dist > radius:
            continue

        fall = 1.0 - (dist / radius)
        fall = fall * fall  # suave
        w = fall * strength
        if w > 1.0:
            w = 1.0

        v.co.z = v.co.z * (1.0 - w) + target_z * w



# -----------------------------
# Operators
# -----------------------------
class RB_OT_create_seed_quad(bpy.types.Operator):
    """Create a small seed quad at 3D cursor to start the road"""
    bl_idname = "rb.create_seed_quad"
    bl_label = "Create Seed Quad"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh object (your tile) first")
            return {'CANCELLED'}

        if context.mode != 'EDIT_MESH':
            bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)
        s = context.scene.rb_settings
        w = s.road_width * 0.5
        length = clamp(s.min_segment, 0.5, s.max_segment)

        c = obj.matrix_world.inverted() @ context.scene.cursor.location

        v0 = bm.verts.new(c + Vector((-w, 0.0, 0.0)))
        v1 = bm.verts.new(c + Vector(( w, 0.0, 0.0)))
        v2 = bm.verts.new(c + Vector(( w, length, 0.0)))
        v3 = bm.verts.new(c + Vector((-w, length, 0.0)))

        try:
            bm.faces.new((v0, v1, v2, v3))
        except ValueError:
            pass

        bm.normal_update()
        bmesh.update_edit_mesh(obj.data)

        # Select front edge (v3-v2)
        front_edge = None
        for e in bm.edges:
            if (e.verts[0] == v3 and e.verts[1] == v2) or (e.verts[0] == v2 and e.verts[1] == v3):
                front_edge = e
                break
        if front_edge:
            set_only_edge_selected(bm, front_edge)
            bmesh.update_edit_mesh(obj.data)

        return {'FINISHED'}


class RB_OT_draw_road(Operator):
    """Draw road by clicking; extrudes currently selected edge as the road 'front'"""
    bl_idname = "rb.draw_road"
    bl_label = "Draw Road"
    bl_options = {'REGISTER', 'UNDO'}

    _bm = None
    _obj = None
    _edge = None
    _last_forward = None
    _color_layer = None

    def invoke(self, context, event):
        obj = ensure_edit_mesh(context)
        if obj is None:
            self.report({'ERROR'}, "You must be in Edit Mode on a Mesh (your tile)")
            return {'CANCELLED'}

        bm = bmesh.from_edit_mesh(obj.data)
        edge = get_selected_edge(bm)
        if edge is None:
            self.report({'ERROR'}, "Select the road front edge (an edge across the road width)")
            return {'CANCELLED'}

        self._obj = obj
        self._bm = bm
        self._edge = edge
        self._last_forward = Vector((0.0, 1.0, 0.0))

        # Ensure vertex color layer (loops)
        s = context.scene.rb_settings
        self._color_layer = ensure_bmesh_color_layer(bm, s.color_layer_name)

        context.window_manager.modal_handler_add(self)
        self.report({'INFO'}, "Road Draw: LMB add segment | RMB/ESC exit")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            bmesh.update_edit_mesh(self._obj.data)
            return {'FINISHED'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            self._add_segment(context, event)
            return {'RUNNING_MODAL'}

        return {'RUNNING_MODAL'}

    def _add_segment(self, context, event):
        s = context.scene.rb_settings
        bm = self._bm
        obj = self._obj
        edge = self._edge
        col_layer = self._color_layer

        # Current edge endpoints (local)
        a0 = edge.verts[0]
        b0 = edge.verts[1]

        # Width axis
        width_vec = (b0.co - a0.co)
        if width_vec.length < 1e-6:
            self.report({'WARNING'}, "Selected edge is too small")
            return False
        width_axis = width_vec.normalized()

        mid0 = (a0.co + b0.co) * 0.5

        # Mouse hit
        hit, loc_w, _ = raycast_mouse_to_scene(context, event)
        if not hit:
            self.report({'WARNING'}, "No hit under mouse (raycast). Aim at the tile mesh.")
            return False

        loc = obj.matrix_world.inverted() @ loc_w

        # Direction towards mouse, projected to keep width constant
        raw = (loc - mid0)
        if raw.length < 1e-6:
            raw = self._last_forward.copy()

        forward = raw - width_axis * raw.dot(width_axis)
        forward = Vector((forward.x, forward.y, 0.0))
        if forward.length < 1e-6:
            forward = self._last_forward.copy()
        forward.normalize()
        self._last_forward = forward.copy()

        # Segment length clamped
        dist = (loc - mid0).length
        seg_len = clamp(dist, s.min_segment, s.max_segment)

        mid1 = mid0 + forward * seg_len

        # Force width to settings
        half_w = s.road_width * 0.5
        a1_pos = mid1 - width_axis * half_w
        b1_pos = mid1 + width_axis * half_w

        # Snap/conform
        if s.snap_to_surface:
            if s.conform_each_vertex:
                ha, la_w, _ = raycast_down(context, obj.matrix_world @ a1_pos)
                hb, lb_w, _ = raycast_down(context, obj.matrix_world @ b1_pos)
                if ha:
                    a1_pos = obj.matrix_world.inverted() @ la_w
                if hb:
                    b1_pos = obj.matrix_world.inverted() @ lb_w
            else:
                # Use the click-hit Z for both
                a1_pos.z = loc.z
                b1_pos.z = loc.z

        # Z offset
        a1_pos.z += s.z_offset
        b1_pos.z += s.z_offset

        # Extrude front edge only (no face)
        res = bmesh.ops.extrude_edge_only(bm, edges=[edge])
        new_verts = [g for g in res['geom'] if isinstance(g, bmesh.types.BMVert)]
        if len(new_verts) != 2:
            self.report({'WARNING'}, "Extrusion did not produce 2 verts (unexpected)")
            return False

        nv0, nv1 = new_verts[0], new_verts[1]
        # Match new verts to old by proximity
        if (nv0.co - a0.co).length <= (nv0.co - b0.co).length:
            a1, b1 = nv0, nv1
        else:
            a1, b1 = nv1, nv0

        a1.co = a1_pos
        b1.co = b1_pos

        # Create the road face (quad)
        f_road = None
        try:
            f_road = bm.faces.new((a0, b0, b1, a1))
        except ValueError:
            # already exists, try find it
            for f in bm.faces:
                vs = set(f.verts)
                if a0 in vs and b0 in vs and a1 in vs and b1 in vs:
                    f_road = f
                    break

        kind = ensure_face_kind_layer(bm)
        set_face_kind(f_road, kind, 1)  # ROAD


        # Paint road face
        road_rgba = hex_to_rgba(s.road_hex, 1.0)
        if f_road:
            paint_face(f_road, col_layer, road_rgba)

        # Flatten terrain under/around the road
        if s.flatten_terrain:
            road_mid = (a0.co + b0.co + a1.co + b1.co) * 0.25
            road_z = road_mid.z
            flatten_terrain_around_segment(
                bm,
                road_mid,
                road_z,
                s.flatten_radius,
                s.flatten_strength
            )

      
        if s.use_knife_flatten and f_road:
            # --------------------------------------------------
            # 1) Guardar ANCLAS de la carretera (antes del knife)
            # --------------------------------------------------
            a0_co = a0.co.copy()
            b0_co = b0.co.copy()
            a1_co = a1.co.copy()
            b1_co = b1.co.copy()

            # --------------------------------------------------
            # 2) Crear cutter del tramo
            # --------------------------------------------------
            cutter = create_segment_cutter_object(
                context, obj,
                a0_co, b0_co, a1_co, b1_co,
                margin=s.knife_margin,
                z_raise=max(0.02, s.z_offset + 0.05)
            )

            # --------------------------------------------------
            # 3) Seleccionar SOLO TERRENO (proteger carretera)
            # --------------------------------------------------
            kind = ensure_face_kind_layer(bm)

            bm.faces.ensure_lookup_table()
            for f in bm.faces:
                f.select = (get_face_kind(f, kind, 0) == 0)  # SOLO TERRAIN
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

            # --------------------------------------------------
            # 4) Knife Project
            # --------------------------------------------------
            knife_project_on_object(
                context,
                obj,
                cutter,
                cut_through=s.knife_cut_through
            )

            bpy.data.objects.remove(cutter, do_unlink=True)

            # --------------------------------------------------
            # 5) REFRESH BMesh (bpy.ops invalida referencias)
            # --------------------------------------------------
            bmesh.update_edit_mesh(obj.data)
            bm = bmesh.from_edit_mesh(obj.data)
            self._bm = bm

            col_layer = ensure_bmesh_color_layer(bm, s.color_layer_name)
            self._color_layer = col_layer

            # --------------------------------------------------
            # 6) Remapear los 4 vértices de la carretera
            # --------------------------------------------------
            a0n = find_vert_near(bm, a0_co)
            b0n = find_vert_near(bm, b0_co)
            a1n = find_vert_near(bm, a1_co)
            b1n = find_vert_near(bm, b1_co)

            if None in (a0n, b0n, a1n, b1n):
                self.report({'WARNING'}, "Knife remap failed")
                return False

            # Usar SOLO verts nuevos a partir de aquí
            a0, b0, a1, b1 = a0n, b0n, a1n, b1n

            # --------------------------------------------------
            # 7) Re-encontrar la cara de carretera en el bm nuevo
            # --------------------------------------------------
            f_road_new = None
            bm.faces.ensure_lookup_table()
            for f in bm.faces:
                vs = set(f.verts)
                if a0 in vs and b0 in vs and a1 in vs and b1 in vs:
                    f_road_new = f
                    break

            # --------------------------------------------------
            # 8) Calcular centro, altura y radio
            # --------------------------------------------------
            mid0 = (a0.co + b0.co) * 0.5
            mid1 = (a1.co + b1.co) * 0.5

            # Carretera lisa → el terreno se adapta a ESTE Z
            target_z = mid0.z

            radius = (s.road_width * 0.5) + s.knife_margin

            # --------------------------------------------------
            # 9) Borrar terreno SOLO si la cara está totalmente dentro
            # --------------------------------------------------
            if s.remove_inner_terrain:
                kind = ensure_face_kind_layer(bm)
                delete_faces_inside_capsule(bm, mid0, mid1, radius, kind_layer=kind, terrain_kind=0)

            # --------------------------------------------------
            # 10) Stitch / Merge terreno → carretera
            # --------------------------------------------------
            if s.stitch_terrain_to_road:

                # Guardar anchors (merge puede matar verts)
                a0_anchor = a0.co.copy()
                a1_anchor = a1.co.copy()
                b0_anchor = b0.co.copy()
                b1_anchor = b1.co.copy()

                snap_and_merge_terrain_to_road_border(
                    bm,
                    a0_anchor, a1_anchor,
                    b0_anchor, b1_anchor,
                    stitch_dist=s.stitch_distance,
                    merge_dist=s.merge_dist,
                    target_z=target_z
                )

                # Remapeo POST-MERGE
                a0 = find_vert_near(bm, a0_anchor)
                a1 = find_vert_near(bm, a1_anchor)
                b0 = find_vert_near(bm, b0_anchor)
                b1 = find_vert_near(bm, b1_anchor)

                if None in (a0, a1, b0, b1):
                    self.report({'WARNING'}, "Post-merge remap failed")
                    return False

            # --------------------------------------------------
            # 11) Flatten del terreno (carretera NO se toca)
            # --------------------------------------------------
            flatten_by_segment_capsule(
                bm,
                mid0,
                mid1,
                target_z,
                radius,
                s.knife_flatten_strength
            )



        # Solidify road (volume) — paint bottom+sides road color too
        if s.solidify_road and f_road:
            created = solidify_face_prism(bm, f_road, s.solid_depth)
            for f in created:
                paint_face(f, col_layer, road_rgba)

        # Curbs (optional)
        if s.add_curbs:
            self._add_curbs_segment(bm, col_layer, a0, a1, b0, b1, width_axis, s)

        # Markings (optional)
        if s.add_markings:
            self._add_markings_segment(bm, col_layer, a0, b0, a1, b1, width_axis, s)

        # Update selection to new front edge (a1-b1)
        new_front = None
        for e in a1.link_edges:
            if b1 in e.verts:
                new_front = e
                break
        if new_front:
            set_only_edge_selected(bm, new_front)
            self._edge = new_front


        bm.normal_update()
        bmesh.update_edit_mesh(obj.data)
        return True

    def _add_curbs_segment(self, bm, col_layer, a0, a1, b0, b1, width_axis, s):
        out_left = (-width_axis).normalized()
        out_right = (width_axis).normalized()
        curb_rgba = hex_to_rgba(s.curb_hex, 1.0)

        def curb_strip(v_start, v_end, outward):
            p0 = v_start.co + outward * s.curb_out + UP * s.curb_up
            p1 = v_end.co   + outward * s.curb_out + UP * s.curb_up
            c0 = bm.verts.new(p0)
            c1 = bm.verts.new(p1)
            face = None
            try:
                face = bm.faces.new((v_start, v_end, c1, c0))
            except ValueError:
                pass
            if face:
                paint_face(face, col_layer, curb_rgba)

        curb_strip(a0, a1, out_left)
        curb_strip(b0, b1, out_right)

    def _add_markings_segment(self, bm, col_layer, a0, b0, a1, b1, width_axis, s):
        m0 = (a0.co + b0.co) * 0.5
        m1 = (a1.co + b1.co) * 0.5
        seg = (m1 - m0)
        if seg.length < 1e-6:
            return
        seg_dir = seg.normalized()

        inset = clamp(s.marking_inset, 0.0, seg.length * 0.45)
        m0i = m0 + seg_dir * inset
        m1i = m1 - seg_dir * inset
        if (m1i - m0i).length < 1e-5:
            return

        half_mw = s.marking_width * 0.5

        left0 = m0i - width_axis * half_mw
        right0 = m0i + width_axis * half_mw
        left1 = m1i - width_axis * half_mw
        right1 = m1i + width_axis * half_mw

        # Raise a bit to avoid z-fighting
        dz = max(0.02, s.z_offset + 0.02)
        left0.z += dz; right0.z += dz; left1.z += dz; right1.z += dz

        v_l0 = bm.verts.new(left0)
        v_r0 = bm.verts.new(right0)
        v_r1 = bm.verts.new(right1)
        v_l1 = bm.verts.new(left1)

        face = None
        try:
            face = bm.faces.new((v_l0, v_r0, v_r1, v_l1))
        except ValueError:
            pass

        if face:
            mark_rgba = hex_to_rgba(s.marking_hex, 1.0)
            paint_face(face, col_layer, mark_rgba)


# -----------------------------
# UI Panel
# -----------------------------
class RB_PT_panel(Panel):
    bl_label = "Road Builder"
    bl_idname = "RB_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Road Builder"

    def draw(self, context):
        layout = self.layout
        s = context.scene.rb_settings

        col = layout.column(align=True)
        col.operator("rb.create_seed_quad", icon="MESH_PLANE")
        col.operator("rb.draw_road", icon="MOD_ARRAY")

        layout.separator()

        box = layout.box()
        box.label(text="Road")
        box.prop(s, "road_width")
        row = box.row(align=True)
        row.prop(s, "min_segment")
        row.prop(s, "max_segment")
        box.prop(s, "snap_to_surface")
        box.prop(s, "conform_each_vertex")
        box.prop(s, "z_offset")

        box = layout.box()
        box.label(text="Vertex Colors (Color Attributes)")
        box.prop(s, "color_layer_name")
        box.prop(s, "road_hex")
        box.prop(s, "marking_hex")
        box.prop(s, "curb_hex")

        box = layout.box()
        box.label(text="Solid Road (no modifier)")
        box.prop(s, "solidify_road")
        sub = box.column(align=True)
        sub.enabled = s.solidify_road
        sub.prop(s, "solid_depth")

        box = layout.box()
        box.label(text="Curbs")
        box.prop(s, "add_curbs")
        sub = box.column(align=True)
        sub.enabled = s.add_curbs
        sub.prop(s, "curb_out")
        sub.prop(s, "curb_up")

        box = layout.box()
        box.label(text="Markings")
        box.prop(s, "add_markings")
        sub = box.column(align=True)
        sub.enabled = s.add_markings
        sub.prop(s, "marking_width")
        sub.prop(s, "marking_inset")


        box = layout.box()
        box.label(text="Knife Flatten (Low Poly)")
        box.prop(s, "use_knife_flatten")
        sub = box.column(align=True)
        sub.enabled = s.use_knife_flatten
        sub.prop(s, "knife_cut_through")
        sub.prop(s, "knife_margin")
        sub.prop(s, "knife_flatten_strength")
        sub.prop(s, "remove_inner_terrain")
        sub.prop(s, "stitch_terrain_to_road")
        sub.prop(s, "stitch_distance")
        sub.prop(s, "merge_road_border")
        sub.prop(s, "merge_dist")



        box = layout.box()
        box.label(text="Terrain Flattening")
        box.prop(s, "flatten_terrain")
        sub = box.column(align=True)
        sub.enabled = s.flatten_terrain
        sub.prop(s, "flatten_radius")
        sub.prop(s, "flatten_strength")
       



# -----------------------------
# Register
# -----------------------------
classes = (
    RB_Settings,
    RB_OT_create_seed_quad,
    RB_OT_draw_road,
    RB_PT_panel,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.rb_settings = PointerProperty(type=RB_Settings)


def unregister():
    del bpy.types.Scene.rb_settings
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
