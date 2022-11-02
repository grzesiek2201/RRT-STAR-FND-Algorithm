import numpy as np
import random
import matplotlib.pyplot as plt
import time
from collections import deque
from collections import defaultdict
import heapq as heap

NODE_RADIUS = 5


class Map:
    def __init__(self, size, start, goal):
        self.obstacles_c = []  # list of obstacles' position and radius (circles)
        self.obstacles_r = []  # list of obstacles' upper-left and lower-right position (rectangles)
        self.width = size[0]
        self.height = size[1]

        self.start = start
        self.goal = goal

        self.space = np.zeros((size[0], size[1]))
        # figure out how to update the plot pls
        self.fig, self.ax = plt.subplots()
        plt.ion()

    def collides_w_start_goal_c(self, obs):
        """ Check if circular obstacle collides with start or goal node """
        distance_start = np.sqrt((obs[0][0] - self.start[0]) ** 2 + (obs[0][1] - self.start[1]) ** 2)
        distance_goal = np.sqrt((obs[0][0] - self.goal[0]) ** 2 + (obs[0][1] - self.goal[1]) ** 2)
        if distance_start < NODE_RADIUS + obs[1] or distance_goal < NODE_RADIUS + obs[1]:
            return True
        return False

    def is_occupied_c(self, p):
        """ Check is space is occupied by circle obstacle """
        for obs in self.obstacles_c:
            distance = np.sqrt((p[0] - obs[0][0])**2 + (p[1] - obs[0][1])**2)
            if distance < NODE_RADIUS + obs[1]:
                return True
        return False

    def generate_obstacles(self, obstacle_count=10, size=1):
        """ Generate random obstacles stored in self.obstacles. Rectangles not yet implemented """
        for _ in range(obstacle_count):
            new_obstacle = (random.randint(0 + size, self.width - size),
                            random.randint(0 + size, self.height - size))
            if self.collides_w_start_goal_c((new_obstacle, size)):
                self.generate_obstacles(obstacle_count=1, size=size)
            else:
                self.obstacles_c.append((new_obstacle, size))

    def show(self):
        """ Show the map """
        plt.gca().set_aspect('equal', adjustable='box')
        for obstacle in self.obstacles_c:
            circle = plt.Circle(obstacle[0], obstacle[1], color='black')
            plt.gca().add_patch(circle)
        plt.show()


class Graph:
    def __init__(self, start, goal, width, height):
        self.start = start  # start position
        self.goal = goal  # goal position
        self.width = width
        self.height = height

        self.vertices = {0: self.start}        # coordinates of vertices retrieved by their id
        self.edges = {0: [(0, 0)]}                     # tuple of id of two vertices connected by an edge

        self.neighbors = {0: []}       # dictionary of list of tuples of neighbors vertices along with cost between them
        self.cost = {0: 0.0}           # dict of distances between two connected vertices
        self.id_vertex = {self.start: 0}    # dict of ids of vertices where key is their position
        self.children = {0: []}
        self.parent = {0: None}

        self.last_id = 0

    def add_vertex(self, pos):
        """ Add new vertex to the Graph """
        try:
            id_vertex = self.id_vertex[pos]
        except KeyError:
            self.last_id += 1
            id_vertex = self.last_id          # id is equal to current length of vertices
            self.vertices[id_vertex] = pos          # add new vertex to vertices list
            self.id_vertex[pos] = id_vertex         # add new vertex's id
            self.neighbors[id_vertex] = []          # add new vertex's neighbors list
            self.children[id_vertex] = []           # add new vertex's children list
            self.edges[id_vertex] = []              # add new vertex's edges list
        return id_vertex

    def remove_vertex(self, id):
        """ Remove vertex from the tree
            id: id of node to remove
        """
        parent = self.parent[id]
        pos = self.vertices[id]
        remove_neighbors(G, id)
        del self.children[parent][self.children[parent].index(id)]
        del self.vertices[id]
        del self.edges[id]
        del self.id_vertex[pos]
        del self.neighbors[id]
        del self.children[id]
        del self.parent[id]
        del self.cost[id]

    def add_edge(self, id1, id2, cost):
        """ Add new edge to the graph """
        self.edges[id1].append((id1, id2))            # append a tuple representing an edge between node id1 and id2
        self.neighbors[id1].append((id2, cost))  # add id2 as neighbor of id2 along with cost between them
        self.neighbors[id2].append((id1, cost))  # add id1 as neighbor of id1 along with cost between them

    def random_node(self, bias=-1):
        """ Generate random point on the map, if bias is between [0;1),
        then there's a probability that point will be the goal point"""
        if random.random() < bias:
            return self.goal
        return random.randint(2, self.width), random.randint(2, self.height)


class Line:
    def __init__(self, p1, p2):
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        self.norm = np.linalg.norm(self.p1 - self.p2)
        if p2[0] - p1[0] == 0:
            self.dir = 0
        else:
            self.dir = (p2[1] - p1[1]) / (p2[0] - p1[0])
        self.const_term = self.p1[1] - self.dir * self.p1[0]

    def calculate_y(self, x0):
        return self.dir * x0 + self.const_term  # y=ax+b -> x = (y-b)/a

    def calculate_x(self, y0):
        return (y0 - self.const_term) / self.dir


def dijkstra(G, final_node):    # room for improvement with PriorityQueue or heapdict
    id_start = G.id_vertex[G.start]
    id_goal = G.id_vertex[G.goal]

    nodes = list(G.neighbors.keys())
    score = {node: float("inf") for node in nodes}
    prev = {node: None for node in nodes}
    score[id_goal] = 0
    while nodes:
        u = min(nodes, key=lambda node: score[node])
        nodes.remove(u)
        if score[u] == float("inf"):
            break
        for v, cost in G.neighbors[u]:
            alt = score[u] + cost
            if alt < score[v]:
                score[v] = alt
                prev[v] = u

    path = deque()
    u = id_start
    while prev[u] is not None:
        path.appendleft(G.vertices[u])
        u = prev[u]
    path.appendleft(G.vertices[u])
    prev_node = G.vertices[id_goal]
    for point in path:
        plt.plot((prev_node[0], point[0]), (prev_node[1], point[1]), c='red')
        prev_node = point
    plt.title(f"Cost: {score[id_start]}")
    return list(path)


def dijkstra2(G, start_node):
    closed = set()                                  # set of visited nodes
    parent = {}                                     # dict of nodes parents
    pq = []
    node_scores = defaultdict(lambda: float("inf"))
    node_scores[G.id_vertex[start_node]] = 0         # set score of start_node to 0
    heap.heappush(pq, (0, G.id_vertex[start_node]))  # add first node to priority queue

    while pq:
        _, min_cost_node = heap.heappop(pq)         # get the node with lowest cost
        if G.vertices[min_cost_node] == G.start:    # if this node is start node, end the loop, solution found
            break
        closed.add(min_cost_node)                   # add selected node to set of visited nodes

        for v, cost in G.neighbors[min_cost_node]:  # iterate through each of the node's neighbors
            if v in closed:                         # if the node is in set of visited nodes, continue
                continue
            alt = node_scores[min_cost_node] + cost  # calculate alternative cost for the node
            if alt < node_scores[v]:                # if the alternative cost is smaller than current
                parent[v] = min_cost_node           # set parent of this node to the min_node
                node_scores[v] = alt                # change the score of this node
                heap.heappush(pq, (alt, v))         # add this node to priority queue to check next

    path = []
    u = G.id_vertex[G.start]                        # add start vertex to path
    while True:
        try:
            path.append(G.vertices[u])              # try to get parent of the node
            u = parent[u]                           # go deeper into the tree
        except KeyError:
            break

    prev_node = G.start
    for point in path:
        plt.plot((prev_node[0], point[0]), (prev_node[1], point[1]), c='red')
        prev_node = point
    plt.plot((prev_node[0], G.goal[0]), (prev_node[1], G.goal[1]), c='red')

    return path, node_scores


if __name__ == '__main__':
    map_width = 200
    map_height = 200
    start_node = (50, 50)
    goal_node = (130, 90)
    G = Graph(start_node, goal_node, map_width, map_height)
    my_map = Map((map_width, map_height), start_node, goal_node)
    my_map.generate_obstacles(obstacle_count=50, size=7)

    iteration = RRT(G, iter_num=200, map=my_map, step_length=25, bias=0)
    plot_graph(G, my_map.obstacles_c)
    print(f"RRT algorithm stopped at iteration number: {iteration}")
    plt.show()

    G = Graph(start_node, goal_node, map_width, map_height)
    iteration = RRT_star(G, iter_num=200, map=my_map, step_length=25, radius=30, bias=0)
    print(f"RRT_star algorithm stopped at iteration number: {iteration}")
    plot_graph(G, my_map.obstacles_c)
    plt.show()

    G = Graph(start_node, goal_node, map_width, map_height)
    iteration = RRT_star_FN(G, iter_num=500, map=my_map, step_length=25, radius=30, max_nodes=20, bias=0)
    print(f"RRT_star_FN algorithm stopped at iteration number: {iteration}")
    plot_graph(G, my_map.obstacles_c)

    plt.show()

    # RRT_STAR robi niepołączone z niczym gałęzie
    # jeden raz przeszedł node przez przeszkodę
    # jest problem z rewire w RRT_STAR