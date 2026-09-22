import cadquery as cq
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Create a cube (box) with dimensions 10x10x10
cube = cq.Workplane("XY").box(10, 10, 10)

# Simple visualization using matplotlib
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# For simplicity, just draw the wireframe of the cube
# Define the 8 vertices of a 10x10x10 cube centered at origin
vertices = np.array([
    [-5, -5, -5], [5, -5, -5], [5, 5, -5], [-5, 5, -5],  # bottom face
    [-5, -5, 5], [5, -5, 5], [5, 5, 5], [-5, 5, 5]       # top face
])

# Define the 12 edges of the cube
edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
    [4, 5], [5, 6], [6, 7], [7, 4],  # top face
    [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
]

# Plot edges
for edge in edges:
    points = vertices[edge]
    ax.plot3D(*points.T, 'b-', linewidth=2)

# Plot vertices
ax.scatter(*vertices.T, c='red', s=50)

# Draw faces with transparency
faces = [
    [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
    [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
    [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
    [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
    [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
    [vertices[1], vertices[2], vertices[6], vertices[5]]   # right
]

face_collection = Poly3DCollection(faces, alpha=0.3, facecolor='cyan', edgecolor='blue')
ax.add_collection3d(face_collection)

# Set labels and title
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('CadQuery Cube (10x10x10)')

# Set equal aspect ratio
max_range = 5
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])
ax.set_zlim([-max_range, max_range])

plt.show()

print("Cube displayed successfully!")
