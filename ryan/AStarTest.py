import edgewiseAStar
import numpy as np
import matplotlib.pyplot as plt

#make a grid
grid = np.zeros((10,10), dtype = bool)

#state a few tiles are occupied
occupied : list[tuple] = [(1,1),(2,1),(4,4),(3,3),(9,3),(4,5),(7,2),(4,3),(3,2),(2,2),(3,5),(3,6)]
#occupied = []
for i in occupied:
    grid[i] = True

#initialize the A* class
AStar : edgewiseAStar.Astar = edgewiseAStar.Astar()

#set the vertices
for xy, val in np.ndenumerate(grid):
    AStar.vertex(xy,val)

#set the edges for the vertices
for (x,y), _ in np.ndenumerate(grid):
    neighbors = [
        (x+dx, y+dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if not (dx == 0 and dy == 0)  # skip the center cell
        if 0 <= x+dx < grid.shape[0] and 0 <= y+dy < grid.shape[1]  # bounds
    ]

    for neighbor in neighbors:
        AStar.edge((x,y), neighbor)

#start and end points, must be tuples
start : tuple = (0,4)
end : tuple = (9,0)
path : list[tuple] = AStar.path(start,end)


#plot the results
if len(occupied):
    nx,ny = zip(*occupied)
    plt.scatter(nx, ny, color = 'red', marker = 'D')
x,y = zip(*path)
plt.scatter(x, y, color='blue', marker='o')
plt.scatter(path[0][0],path[0][1],color = 'lightgreen', marker = '*')
plt.scatter(x = path[-1][0], y = path[-1][1], color='yellow',marker='*')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.xlim(-1,10)
plt.ylim(-1,10)
plt.xticks(range(0, 11))
plt.yticks(range(0, 11))
plt.title('Best Path from ' + str(start) + ' to ' +  str(end) + '\n' + (' with no Obstacles' if not len(occupied) else ('with Obstacles at' + str(occupied))))

ax.grid(True, 'both')
plt.show()