import numpy as np
import matplotlib.pyplot as plt
import random
import math

# CONFIGURATION

ROLL_NUMBER = "01136232076" 
SEED_VAL = int(''.join(filter(str.isdigit, ROLL_NUMBER)))

GRID_SIZE = 20
OBSTACLE_RATIO = 0.25 
NUM_WAYPOINTS = 3     

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def generate_environment(size, obstacle_ratio):
    grid = np.zeros((size, size))
    num_obstacles = int(size * size * obstacle_ratio)
    
    obstacles = set()
    while len(obstacles) < num_obstacles:
        x, y = random.randint(0, size-1), random.randint(0, size-1)
        obstacles.add((x, y))
        grid[y, x] = 1 

    start = (random.randint(0, size-1), random.randint(0, size-1))
    while start in obstacles:
        start = (random.randint(0, size-1), random.randint(0, size-1))
        
    goal = (random.randint(0, size-1), random.randint(0, size-1))
    while goal in obstacles or goal == start:
        goal = (random.randint(0, size-1), random.randint(0, size-1))

    return grid, list(obstacles), start, goal

def get_line_cells(x0, y0, x1, y1):
    cells = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            cells.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            cells.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    cells.append((x, y))
    return cells

def calculate_fitness(particle, start, goal, obstacles):
    waypoints = [start]
    for i in range(0, len(particle), 2):
        waypoints.append((particle[i], particle[i+1]))
    waypoints.append(goal)
    
    total_distance = 0
    penalty = 0
    
    for i in range(len(waypoints) - 1):
        x0, y0 = int(round(waypoints[i][0])), int(round(waypoints[i][1]))
        x1, y1 = int(round(waypoints[i+1][0])), int(round(waypoints[i+1][1]))
        
        total_distance += math.hypot(x1 - x0, y1 - y0)
        
        cells = get_line_cells(x0, y0, x1, y1)
        for cell in cells:
            if cell in obstacles:
                penalty += 1000 
            if cell[0] < 0 or cell[0] >= GRID_SIZE or cell[1] < 0 or cell[1] >= GRID_SIZE:
                penalty += 1000 
                
    return total_distance + penalty

def run_pso(start, goal, obstacles):
    NUM_PARTICLES = 100
    MAX_ITER = 150
    W, C1, C2 = 0.7, 1.5, 1.5
    dim = NUM_WAYPOINTS * 2
    
    particles = np.random.uniform(0, GRID_SIZE-1, (NUM_PARTICLES, dim))
    velocities = np.random.uniform(-1, 1, (NUM_PARTICLES, dim))
    
    pbest = np.copy(particles)
    pbest_fitness = np.array([calculate_fitness(p, start, goal, obstacles) for p in particles])
    
    gbest = pbest[np.argmin(pbest_fitness)]
    gbest_fitness = np.min(pbest_fitness)
    
    for iter in range(MAX_ITER):
        for i in range(NUM_PARTICLES):
            r1, r2 = random.random(), random.random()
            
            velocities[i] = (W * velocities[i] + 
                             C1 * r1 * (pbest[i] - particles[i]) + 
                             C2 * r2 * (gbest - particles[i]))
            
            particles[i] = particles[i] + velocities[i]
            particles[i] = np.clip(particles[i], 0, GRID_SIZE-1)
            
            fitness = calculate_fitness(particles[i], start, goal, obstacles)
            
            if fitness < pbest_fitness[i]:
                pbest_fitness[i] = fitness
                pbest[i] = particles[i]
                
        current_gbest_idx = np.argmin(pbest_fitness)
        if pbest_fitness[current_gbest_idx] < gbest_fitness:
            gbest_fitness = pbest_fitness[current_gbest_idx]
            gbest = np.copy(pbest[current_gbest_idx])
            
        if iter % 25 == 0:
            print(f"Iteration {iter}: Best Cost = {gbest_fitness:.2f}")
            
    return gbest, gbest_fitness

def plot_environment(grid, start, goal, best_path=None):
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap='Greys', origin='lower')
    
    plt.scatter(*start, color='green', s=150, marker='s', label='Start')
    plt.scatter(*goal, color='red', s=150, marker='*', label='Goal')
    
    if best_path is not None:
        path_x = [start[0]] + [best_path[i] for i in range(0, len(best_path), 2)] + [goal[0]]
        path_y = [start[1]] + [best_path[i+1] for i in range(0, len(best_path), 2)] + [goal[1]]
        plt.plot(path_x, path_y, color='blue', linewidth=3, marker='o', markersize=6, label='PSO Path')

    plt.grid(True, which='both', color='lightgrey', linewidth=0.5)
    plt.xticks(np.arange(-0.5, GRID_SIZE, 1), [])
    plt.yticks(np.arange(-0.5, GRID_SIZE, 1), [])
    plt.title(f"PSO Pathfinding (Seed/Roll No: {ROLL_NUMBER})")
    plt.legend()
    plt.show()

# This is the execution block that actually runs the code
if __name__ == "__main__":
    print(f"Initializing Environment for Roll Number: {ROLL_NUMBER}")
    set_seed(SEED_VAL)
    
    grid, obstacles, start, goal = generate_environment(GRID_SIZE, OBSTACLE_RATIO)
    print(f"Start Point: {start} | Goal Point: {goal}")
    
    print("Running Particle Swarm Optimization...")
    best_particle, cost = run_pso(start, goal, obstacles)
    
    print(f"\nFinal Best Path Cost (Length + Penalties): {cost:.2f}")
    plot_environment(grid, start, goal, best_particle)