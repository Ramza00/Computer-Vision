import edgewiseAStar
import json
import numpy as np

with open("astar_grid_test.json", "r") as f:
    data = json.load(f)

for test in data:
    print("testing ", test["name"])
    grid = np.array(test["grid"])
    start = tuple(test['start'])
    goal = tuple(test['goal'])
    expected = test['expected']
    
    AStar : edgewiseAStar.Astar = edgewiseAStar.Astar()

    #set the vertices
    for xy, val in np.ndenumerate(grid):
        AStar.vertex(xy,val)

    #set the edges for the vertices
    for (x,y), _ in np.ndenumerate(grid):
        neighbors = [(x-1,y), (x+1,y), (x,y-1), (x,y+1)]
        neighbors = [(nx,ny) for nx,ny in neighbors if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]]

        for neighbor in neighbors:
            AStar.edge((x,y), neighbor)

    result = AStar.path(start,goal)
    
    #abuse falsey to print correct results
    if len(result): result.pop(0)
    print("Expected: ", expected,"\nResult: ", str(len(result)) + "\nSpecific path: " + str(result) if len(result) else False)