import matplotlib.pyplot as plt
import numpy as np

# Grid parameters
grid_size = 500
steps = 100

# Initialize nutrient grid - each cell randomly assigned one nutrient type
# 0 = nitrogen, 1 = phosphorus, 2 = glucose
nutrient_grid = np.random.randint(0, 3, size=(grid_size, grid_size))

# Create color grid for visualization
def create_nutrient_image(nutrient_grid):
    image = np.zeros((grid_size, grid_size, 3))
    # Nitrogen = light green
    nitrogen_mask = nutrient_grid == 0
    image[nitrogen_mask] = [0.7, 0.95, 0.7]
    
    # Phosphorus = light pink
    phosphorus_mask = nutrient_grid == 1
    image[phosphorus_mask] = [1.0, 0.8, 0.9]
    
    # Glucose = orange
    glucose_mask = nutrient_grid == 2
    image[glucose_mask] = [1.0, 0.7, 0.4]
    
    return image

# Initialize mycelium grid
mycelium = np.zeros((grid_size, grid_size))
# Randomly disperse 30 spores
num_spores = 30
spore_positions = []
spore_directions = []
for _ in range(num_spores):
    spore_y = np.random.randint(0, grid_size)
    spore_x = np.random.randint(0, grid_size)
    mycelium[spore_y, spore_x] = 1
    spore_positions.append((spore_y, spore_x))
    # Assign each spore a random direction
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    spore_directions.append(directions[np.random.randint(0, len(directions))])

# Growth function with random walk in assigned direction per spore
def grow_mycelium(mycelium, nutrient_grid, spore_positions, spore_directions, steps=50):
    # Track the tip of each filament
    filament_tips = [(y, x, dy, dx) for (y, x), (dy, dx) in zip(spore_positions, spore_directions)]
    
    for step in range(steps):
        new_tips = []
        
        for y, x, dy, dx in filament_tips:
            # Random walk: occasionally change direction slightly
            if np.random.rand() < 0.1:  # 10% chance to adjust direction
                # Small random adjustment
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
                dy, dx = directions[np.random.randint(0, len(directions))]
            
            # Try to grow in the assigned direction
            ny, nx = y + dy, x + dx
            
            # Check bounds
            if 0 <= ny < grid_size and 0 <= nx < grid_size:
                # Check if already occupied
                if mycelium[ny, nx] == 0:
                    # Growth probability based on nutrient type
                    nutrient_type = nutrient_grid[ny, nx]
                    if nutrient_type == 0:  # Nitrogen - high preference
                        growth_prob = 0.8
                    elif nutrient_type == 1:  # Phosphorus - high preference
                        growth_prob = 0.8
                    else:  # Glucose - lower preference
                        growth_prob = 0.3
                    
                    if np.random.rand() < growth_prob:
                        mycelium[ny, nx] = 1
                        new_tips.append((ny, nx, dy, dx))
                    else:
                        # Keep trying from the same position
                        new_tips.append((y, x, dy, dx))
                else:
                    # Keep trying from the same position
                    new_tips.append((y, x, dy, dx))
            else:
                # Hit boundary, stop this filament
                pass
        
        filament_tips = new_tips
        
        if len(filament_tips) == 0:
            break
    
    return mycelium

# Grow mycelium
mycelium = grow_mycelium(mycelium, nutrient_grid, spore_positions, spore_directions, steps=steps)

# Create visualization
nutrient_image = create_nutrient_image(nutrient_grid)

# Create combined image with mycelium overlay
combined = nutrient_image.copy()
mycelium_mask = mycelium > 0
combined[mycelium_mask] = [0.05, 0.05, 0.05]  # Dark mycelium

# Display results
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot nutrients only
axes[0].imshow(nutrient_image, interpolation='nearest')
axes[0].set_title('Random Nutrient Distribution\n(Green=Nitrogen, Pink=Phosphorus, Orange=Glucose)', fontsize=14)
axes[0].axis('off')

# Plot with mycelium
axes[1].imshow(combined, interpolation='nearest')
axes[1].set_title('Mycelium Growth (favoring Nitrogen & Phosphorus)', fontsize=14)
axes[1].axis('off')

plt.tight_layout()
plt.show()

# Create a larger detailed view
fig2, ax = plt.subplots(figsize=(12, 12))
ax.imshow(combined, interpolation='nearest')
ax.set_title('Branched Mycelium Network on Nutrient Grid', fontsize=16)
ax.axis('off')
plt.tight_layout()
plt.show()
