import cadquery as cq
import numpy as np

# Set random seed for reproducibility (optional)
np.random.seed(42)

#number of pores to generate
num_pores = 2000

# Create a cube (box) with dimensions 10x10x10
cube = cq.Workplane("XY").box(10, 10, 10)

# Generate random position within the cube
# Cube is centered at origin, so ranges from -5 to 5 in each dimension
sphere_radius = 0.2

# Create all spheres first and combine them
print("Generating spheres...")
all_spheres = None

for i in range(num_pores):
    x = np.random.uniform(-5, 5)
    y = np.random.uniform(-5, 5)
    z = np.random.uniform(-5, 5)
    sphere = cq.Workplane("XY").center(x, y).sphere(sphere_radius).translate((0, 0, z))
    
    if all_spheres is None:
        all_spheres = sphere
    else:
        all_spheres = all_spheres.union(sphere)
    
    if (i + 1) % 50 == 0:
        print(f'Generated {i+1}/{num_pores} spheres')

# Now cut all spheres from the cube in ONE operation
print("Cutting all spheres from cube...")
result = cube.cut(all_spheres)

# Display the result
if __name__ == "__main__":
    # Export to STEP file for viewing in external CAD software
    print("Exporting to STEP file...")
    cq.exporters.export(result, "cube_with_holes.step")
    print(f"Porous cube with {num_pores} holes created")
    print(f"Sphere radius: {sphere_radius}")
    print("Result exported to 'cube_with_holes.step'")
    print("You can view it with FreeCAD or any STEP viewer")
    
    # Alternatively, try to show in browser using cq-server if installed
    try:
        from jupyter_cadquery import show
        show(result)
    except ImportError:
        print("For interactive viewing, install: pip install jupyter-cadquery")
        print("Then run this script in Jupyter notebook")
