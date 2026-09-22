import cadquery as cq
import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCP.gp import gp_Pnt, gp_Trsf, gp_Vec
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform

# Set random seed for reproducibility (optional)
np.random.seed(42)

# Number of pores to generate
num_pores = 800
sphere_radius = 0.8

# Create a cube (box) with dimensions 10x10x10
print("Creating cube...")
cube = cq.Workplane("XY").box(10, 10, 10)
result_shape = cube.val().wrapped

# Generate spheres and cut them in batches for better performance
batch_size = 50
print(f"Generating and cutting {num_pores} spheres in batches of {batch_size}...")

for batch_start in range(0, num_pores, batch_size):
    batch_end = min(batch_start + batch_size, num_pores)
    
    # Create spheres for this batch
    spheres_in_batch = []
    for i in range(batch_start, batch_end):
        x = np.random.uniform(-5, 5)
        y = np.random.uniform(-5, 5)
        z = np.random.uniform(-5, 5)
        
        # Create sphere at origin
        sphere = BRepPrimAPI_MakeSphere(sphere_radius).Shape()
        
        # Translate to position
        transformation = gp_Trsf()
        transformation.SetTranslation(gp_Vec(x, y, z))
        transformed_sphere = BRepBuilderAPI_Transform(sphere, transformation, True).Shape()
        
        spheres_in_batch.append(transformed_sphere)
    
    # Fuse all spheres in this batch
    if len(spheres_in_batch) > 0:
        fused_batch = spheres_in_batch[0]
        for sphere in spheres_in_batch[1:]:
            fuser = BRepAlgoAPI_Fuse(fused_batch, sphere)
            fuser.Build()
            fused_batch = fuser.Shape()
        
        # Cut the fused batch from the result
        cutter = BRepAlgoAPI_Cut(result_shape, fused_batch)
        cutter.Build()
        result_shape = cutter.Shape()
    
    print(f'Processed {batch_end}/{num_pores} spheres')

# Wrap back into CadQuery object
result = cq.Shape.cast(result_shape)

# Export
print("Exporting to STEP file...")
cq.exporters.export(result, "cube_with_holes_fast.step")
print(f"Porous cube with {num_pores} holes created")
print(f"Sphere radius: {sphere_radius}")
print("Result exported to 'cube_with_holes_fast.step'")
