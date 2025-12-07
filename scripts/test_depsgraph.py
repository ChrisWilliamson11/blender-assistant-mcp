import bpy

def on_depsgraph_update(scene, depsgraph):
    # Iterate over updates
    print(f"\n[Depsgraph Update] Scene: {scene.name}")
    for update in depsgraph.updates:
        print(f"  - Update: {update.id.name} (Type: {type(update.id)})")
        if update.is_updated_geometry:
            print("    -> Geometry Updated")
        if update.is_updated_transform:
            print("    -> Transform Updated")
        if update.is_updated_shading:
            print("    -> Shading Updated")

# clear old
bpy.app.handlers.depsgraph_update_post.clear()
bpy.app.handlers.depsgraph_update_post.append(on_depsgraph_update)

print("Started monitoring...")

# Test Actions
print("Adding Cube...")
bpy.ops.mesh.primitive_cube_add(location=(10, 10, 10))
bpy.context.view_layer.update() # Force update

print("Moving Cube...")
bpy.context.object.location.x += 1
bpy.context.view_layer.update()

print("Changing Material...")
mat = bpy.data.materials.new(name="TestMat")
bpy.context.object.data.materials.append(mat)
bpy.context.view_layer.update()

print("Cleaning up...")
bpy.app.handlers.depsgraph_update_post.remove(on_depsgraph_update)
