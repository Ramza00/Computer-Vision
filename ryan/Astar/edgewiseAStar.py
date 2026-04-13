import heapq

#Heuristical Astar
class Astar:
    def __init__(self):
        #the vertex hashmap, recording the xy coordinate and occupation as the key:value pairing.
        #This can be later adjusted using truthy to account for variadic cost and occupation.
        #Probably would need to invert the occupation when this is done to use 0 as falsey.
        self.vMap : dict[tuple,bool] = {} 
        #The edge hashmap, recording the two connected xy coordinates.
        self.eMap : dict[tuple,list[tuple]] = {}
    
    #Set a vertex at the indicated xy coordinates. Occupation indicates inability to traverse.
    def vertex(self, xy: tuple, occupied = False) -> None:
        #default occupation of the vertex is False (no obstacle)
        self.vMap[xy] = occupied
        self.eMap[xy] = []

    #Set the edge along two neighboring vertices.
    def edge(self, xoyo: tuple, xy: tuple) -> None:
        #automatic establishing of vertices if not already set
        if not xoyo in self.vMap:
            self.vertex(xoyo)
        if not xy in self.vMap:
            self.vertex(xy)
        #record the edges -> this is hard coded as bidirectional, because of the dynamic nature of the map.
        self.eMap[xoyo].append(xy)
        self.eMap[xy].append(xoyo)

    #Return the heuristic for two points
    def heuristic(self, xyo: tuple, xy: tuple) -> float:
        xo, yo = xyo
        x, y = xy
        d = ((x - xo)**2 + (y - yo)**2)**0.5
        return d
    
    #Return the best path of from the current position to the goal. If no path available, or already at goal, returns an empty array.
    #The path considers the originating position (position) as part of the path.
    def path(self, position: tuple, goal: tuple) -> list[tuple]:
        #establish a priority queue (heap) and also another hashmap that details what vertices have been used.
        pq, used = [], {}
        #initialize heap with current position
        #first is the cost of the vertex, second is the cumulative g, third is the current position, and fourth is the path via array.
        heapq.heappush(pq, (0, 0, position, [position]))

        #while there are edges left to be evaluated
        while pq:
            #Use the priority queue to get the position of most potential (least cost + heuristic)
            _, g, current, path = heapq.heappop(pq)

            #if arrived at goal
            if current == goal:
                return path

            #otherwise, collect the neighbors of the position
            neighbors = self.eMap[current]

            #then, for every neighbor, evaluate the local distance cost and heuristic cost and push them to the priority queue.
            for neighbor in neighbors:
                #Change when cost is variable
                #if occupied, skip this neighbor
                if self.vMap[neighbor]:
                    continue

                #consider the path. If the vertex that it attempting to be traveled to has already been traveled, because of the uniform cost,
                #the new path is guaranteed to be more expensive, so therefore remove it.
                #when making a non-uniform plane, we can consider g+dg and compare it to a new hashmap that contains used vertices and their cost.
                if neighbor in used:
                    continue
                used[neighbor] = True

                #add the neighbor to the path
                nPath = path + [neighbor]

                #calculate the new edge cost -> just 1.0 per edge
                #you can reuse the heuristic cost to calculate physical g in next iterations
                dg = 1.0

                #print("considering", neighbor)
                #the new cost is the cumulative g plus new edge cost plus the heuristic
                cost = g+dg + self.heuristic(neighbor,goal)
                #push the item to the priority queue
                heapq.heappush(pq, (cost, g+dg, neighbor, nPath))

        return []
    
    #Reduce the path. Simplistic implementation made by removing any points in a line segment.
    #Expects the result of the path method as an input.
    #If a reduced path without the origin is needed, use reduce(path())[1:]
    def reduce(self,path: list[tuple]) -> list[tuple]:
        if len(path) < 2:
            return path

        newpath: list[tuple] = [path[0]]

        def direction(a: tuple, b: tuple) -> tuple:
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            return (dx // abs(dx) if dx else 0,
                    dy // abs(dy) if dy else 0)

        current_dir = direction(path[0], path[1])

        for i in range(1, len(path) - 1):
            next_dir = direction(path[i], path[i + 1])
            if next_dir != current_dir:
                newpath.append(path[i])
                current_dir = next_dir

        newpath.append(path[-1])
        return newpath
