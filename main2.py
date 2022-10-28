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

        self.parent = {0: None}
        self.children = {0: []}
        self.vertices = {0: self.start}
        self.cost = {0: 0.0}
        self.id_vertex = {self.start: 0}

        # self.vertices = {0: self.start}        # coordinates of vertices retrieved by their id
        # self.edges = {0: [(0, 0)]}                     # tuple of id of two vertices connected by an edge

        # self.neighbors = {0: []}       # dictionary of list of tuples of neighbors vertices along with cost between them
        # self.cost = {0: 0.0}           # dict of distances between two connected vertices
        # self.id_vertex = {self.start: 0}    # dict of ids of vertices where key is their position
        # self.children = {0: []}
        # self.parent = {0: None}

        self.last_id = 0    # ?

    def add_vertex(self, pos):
        """ Add new vertex to the Graph """
        try:
            id_vertex = self.id_vertex[pos]
        except KeyError:
            self.last_id += 1
            id_vertex = self.last_id          # id is equal to current length of vertices
            self.vertices[id_vertex] = pos          # add new vertex to vertices list
            self.id_vertex[pos] = id_vertex         # add new vertex's id
            self.children[id_vertex] = []           # add new vertex's children list
            # self.parent[id_vertex] = None         # not sure if initialization needed
            # self.cost[id_vertex] = float("inf")   # not sure if initialization needed
        return id_vertex

    def remove_vertex(self, id):
        """ Remove vertex from the tree
            id: id of node to remove
        """
        parent = self.parent[id]
        pos = self.vertices[id]
        # remove_neighbors(G, id)
        del self.parent[id]
        del self.children[parent][self.children[parent].index(id)]
        del self.children[id]
        del self.vertices[id]
        del self.cost[id]
        del self.id_vertex[pos]
        # del self.edges[id]
        # del self.neighbors[id]

    def add_edge(self, id_node, id_parent, cost):
        """ Add new edge to the graph """
        self.parent[id_node] = id_parent
        self.children[id_parent].append(id_node)

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


def remove_neighbors(G, id):
    connected_nodes = [node for node, cost in G.neighbors[id]]  # nodes that have id node as a neighbor
    for node in connected_nodes:
        temp_id = -1
        for neighbor in G.neighbors[node]:                      # iterate through neighbors of one of connected nodes
            temp_id += 1                                        # find index of item that corresponds to id
            if neighbor[0] == id:
                break
        del G.neighbors[node][temp_id]                          # delete item that has the id from the list


def calc_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def intersection_circle(line, circle):
    """ Check for intersection between a section and a circle """
    p1, p2 = line.p1, line.p2
    delta, a, b = calc_delta(line, circle)
    if delta < 0:   # no real solutions
        return False
    ip1, ip2 = delta_solutions(delta, a, b)  # intersection point 1's x; intersection point 2's x
    ip1[1] = line.calculate_y(ip1[0])
    ip2[1] = line.calculate_y(ip2[0])
    if (ip1[0] < 0 and ip2[0] < 0) or (ip1[1] < 0 and ip2[1] < 0):  # outside the map
        return False
    if is_between(ip1, p1, p2) or is_between(ip2, p1, p2):
        return True
    return False


def through_obstacle(line, obstacles):  # only for circular obstacles for now
    for obstacle in obstacles:
        if intersection_circle(line, obstacle):
            return True
    return False


def delta_solutions(delta, a, b):
    """ Calculate solutions from delta and quadratic equation's coefficients """
    x1 = [(-b + delta**0.5) / (2 * a), None]
    x2 = [(-b - delta**0.5) / (2 * a), None]
    return x1, x2


def calc_delta(line, circle):
    """ Calculate delta, as well as equation's coefficients and return them """
    x0 = circle[0][0]  # circle's center x coordinate
    y0 = circle[0][1]  # circle's center y coordinate
    r = circle[1]      # circle's radius
    a = (1 + line.dir**2)
    b = 2 * (-x0 + line.dir * line.const_term - line.dir * y0)
    c = -r ** 2 + x0 ** 2 + y0 ** 2 - 2 * y0 * line.const_term + line.const_term ** 2
    delta = b ** 2 - 4 * a * c
    return delta, a, b


def is_between(pb, p1, p2):
    """ Check if pb lies on a section on (p1, p2) """
    check1 = pb[0] > min(p1[0], p2[0])
    check2 = pb[0] < max(p1[0], p2[0])

    return check1 and check2


def plot_graph(graph, obstacles):
    xes = [pos[0] for id, pos in graph.vertices.items()]
    yes = [pos[1] for id, pos in graph.vertices.items()]

    plt.scatter(xes, yes)   # plotting nodes
    plt.scatter(graph.start[0], graph.start[1], c='pink')
    plt.scatter(graph.goal[0], graph.goal[1], c='red')

    edges = [(graph.vertices[id_ver], graph.vertices[child]) for pos_ver, id_ver in graph.id_vertex.items()
             for child in graph.children[id_ver]]
    for edge in edges:
        plt.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], c='black', alpha=0.5)

    # plot obstacles
    plt.gca().set_aspect('equal', adjustable='box')
    for obstacle in obstacles:
        circle = plt.Circle(obstacle[0], obstacle[1], color='black')
        plt.gca().add_patch(circle)

    plt.xlim(0, 200)
    plt.ylim(0, 200)


def nearest_node(graph, vertex, obstacles):
    """ Checks for the nearest node to the input node, check for crossing obstacles,
     returns: new_vertex, new_id """

    try:
        id = graph.id_vertex[vertex]
        return vertex, id
    except KeyError:
        min_distance = float("inf")
        new_id = None
        new_vertex = None
        for ver_id, ver in graph.vertices.items():      # was enumerate(graph.vertices)
            line = Line(ver, vertex)
            if through_obstacle(line, obstacles):
                continue
            distance = calc_distance(ver, vertex)
            if distance < min_distance:
                min_distance = distance
                new_id = ver_id
                new_vertex = ver

        return new_vertex, new_id


def new_node(to_vertex, from_vertex, max_length):
    """ Return position of new node. from_vertex -> to_vertex, of given length"""
    distance = calc_distance(to_vertex, from_vertex)
    x_vect_norm = (to_vertex[0] - from_vertex[0]) / distance
    y_vect_norm = (to_vertex[1] - from_vertex[1]) / distance
    x_pos = from_vertex[0] + x_vect_norm * max_length
    y_pos = from_vertex[1] + y_vect_norm * max_length
    if distance > max_length:
        return x_pos, y_pos
    return to_vertex


def RRT(G, iter_num, map, step_length, bias=.0):
    """ RRT algorithm """
    obstacles = map.obstacles_c
    # goal_node = None
    iter = 0
    while iter < iter_num:
        q_rand = G.random_node(bias=bias)                           # generate a new random node
        if map.is_occupied_c(q_rand):                               # if new node's position is in an obstacle
            continue
        q_near, id_near = nearest_node(G, q_rand, obstacles)        # search for the nearest node
        if q_near is None:                                 # random node cannot be connected to nearest without obstacle
            continue
        q_new = new_node(q_rand, q_near, step_length)
        id_new = G.add_vertex(q_new)
        distance = calc_distance(q_new, q_near)
        G.add_edge(id_new, id_near, distance)
        # plot_graph(G, map.obstacles_c)
        # plt.show()

        dist_to_goal = calc_distance(q_new, G.goal)
        if dist_to_goal < 2 * NODE_RADIUS:
            # G.add_vertex(G.goal)
            # G.add_edge(id_new, G.id_vertex[G.goal], dist_to_goal)
            print("********PATH FOUND**********")
            # dijkstra2(G, q_new)
            break

        iter += 1

    return iter


def find_best_node(G, q_new, id_new, best_edge, radius, obstacles):
    for id_ver, vertex in G.vertices.items():  # iterate through all the vertices
        if id_ver == id_new:
            continue
        distance_new_vert = calc_distance(q_new, vertex)  # calculate distance between new node and vertex node
        if distance_new_vert > radius:  # if distance is greater than search radius - continue
            continue
        line = Line(vertex, q_new)  # create Line object from new node to vertex
        if through_obstacle(line, obstacles):  # if the line goes through obstacle - continue
            continue
        if G.cost[id_new] > G.cost[id_ver] + distance_new_vert:  # if cost from new node to vertex is smaller
            G.cost[id_new] = G.cost[id_ver] + distance_new_vert  # than current cost, rewire the vertex to new
            best_edge = (id_new, id_ver, distance_new_vert)

    return best_edge


def rewire(G, q_new, id_new, radius, obstacles):
    for id_ver, vertex in G.vertices.items():  # iterate through all the vertices
        if id_ver == id_new:
            continue
        distance_new_vert = calc_distance(q_new, vertex)  # calculate distance between new node and vertex node
        if distance_new_vert > radius:  # if distance is greater than search radius - continue
            continue
        line = Line(vertex, q_new)  # create Line object from new node to vertex
        if through_obstacle(line, obstacles):  # if the line goes through obstacle - continue
            continue
        if G.cost[id_new] + distance_new_vert < G.cost[id_ver]:  # if cost from new node to vertex is smaller
            # G.cost[id_ver] = G.cost[id_new] + distance_new_vert
            # G.edges[id_ver] = [(id_ver, id_new)]    # PROBLEM HERE *******************************************
            parent = G.parent[id_ver]
            del G.children[parent][G.children[parent].index(id_ver)]
            G.parent[id_ver] = id_new
            G.children[id_new].append(id_ver)
            G.cost[id_ver] = G.cost[id_new] + distance_new_vert
            G.edges[id_ver] = [(id_ver, id_new)]      # PROBLEM HERE *******************************************


def check_solution(G, q_new, id_new):
    dist_to_goal = calc_distance(q_new, G.goal)  # check if the goal has been reached
    if dist_to_goal < 2 * NODE_RADIUS:
        G.add_vertex(G.goal)
        G.add_edge(id_new, G.id_vertex[G.goal], dist_to_goal)
        print("********PATH FOUND**********")
        dijkstra2(G, q_new)
        return True  # it should absolutely not stop here, but instead continue to grow the tree and search for optimal solution
    return False


def RRT_star(G, iter_num, map, step_length, radius, bias=.0):
    """ RRT star algorithm """
    obstacles = map.obstacles_c
    best_edge = None
    iter = 0
    while iter < iter_num:
        q_rand = G.random_node(bias=bias)   # generate random node
        if map.is_occupied_c(q_rand):       # if it's generated on an obstacle, continue
            continue
        q_near, id_near = nearest_node(G, q_rand, obstacles)    # find the nearest to the random node; change function to also include radius?
        if q_near is None:
            continue
        q_new = new_node(q_rand, q_near, step_length)       # create the new node
        id_new = G.add_vertex(q_new)                        # get id of the new node
        cost_new_near = calc_distance(q_new, q_near)        # find cost from q_new to q_near
        best_edge = (id_new, id_near, cost_new_near)
        G.cost[id_new] = G.cost[id_near] + cost_new_near    # calculate cost for new node from nearest node
        # plot_graph(G, map.obstacles_c)
        # plt.show()
        # for id_ver, vertex in G.vertices.items():                           # iterate through all the vertices
        #     if id_ver == id_new:
        #         continue
        #     distance_new_vert = calc_distance(q_new, vertex)    # calculate distance between new node and vertex node
        #     if distance_new_vert > radius:                      # if distance is greater than search radius - continue
        #         continue
        #     line = Line(vertex, q_new)                          # create Line object from new node to vertex
        #     if through_obstacle(line, obstacles):               # if the line goes through obstacle - continue
        #         continue
        #     if G.cost[id_new] > G.cost[id_ver] + distance_new_vert:  # if cost from new node to vertex is smaller
        #         G.cost[id_new] = G.cost[id_ver] + distance_new_vert  # than current cost, rewire the vertex to new
        #         best_edge = (id_new, id_ver, distance_new_vert)
        find_best_node(G, q_new, id_new, best_edge, radius, obstacles)

        G.parent[id_new] = best_edge[1]
        G.children[best_edge[1]].append(id_new)
        G.add_edge(*best_edge)

        # rewire
        rewire(G, q_new, id_new, radius, obstacles)
        # for id_ver, vertex in G.vertices.items():  # iterate through all the vertices
        #     if id_ver == id_new:
        #         continue
        #     distance_new_vert = calc_distance(q_new, vertex)  # calculate distance between new node and vertex node
        #     if distance_new_vert > radius:  # if distance is greater than search radius - continue
        #         continue
        #     line = Line(vertex, q_new)  # create Line object from new node to vertex
        #     if through_obstacle(line, obstacles):  # if the line goes through obstacle - continue
        #         continue
        #     if G.cost[id_new] + distance_new_vert < G.cost[id_ver]:  # if cost from new node to vertex is smaller
        #         G.cost[id_ver] = G.cost[id_new] + distance_new_vert
        #         G.edges[id_ver] = [(id_ver, id_new)]
        # dist_to_goal = calc_distance(q_new, G.goal)                     # check if the goal has been reached
        # if dist_to_goal < 2 * NODE_RADIUS:
        #     G.add_vertex(G.goal)
        #     G.add_edge(id_new, G.id_vertex[G.goal], dist_to_goal)
        #     print("********PATH FOUND**********")
        #     dijkstra2(G, q_new)
        #     break       # it should absolutely not stop here, but instead continue to grow the tree and search for optimal solution
        if check_solution(G, q_new, id_new):
            break
        iter += 1

    return iter


def RRT_star_FN(G, iter_num, map, step_length, radius, max_nodes=200, bias=.0):
    """ RRT star algorithm """
    obstacles = map.obstacles_c
    best_edge = None
    n_of_nodes = 1  # only starting node at the beginning
    iter = 0
    while iter < iter_num:
        q_rand = G.random_node(bias=bias)   # generate random node
        if map.is_occupied_c(q_rand):       # if it's generated on an obstacle, continue
            continue
        q_near, id_near = nearest_node(G, q_rand, obstacles)    # find the nearest to the random node; change function to also include radius?
        if q_near is None:
            continue
        q_new = new_node(q_rand, q_near, step_length)       # create the new node
        id_new = G.add_vertex(q_new)                        # get id of the new node
        cost_new_near = calc_distance(q_new, q_near)        # find cost from q_new to q_near
        best_edge = (id_new, id_near, cost_new_near)
        G.cost[id_new] = G.cost[id_near] + cost_new_near    # calculate cost for new node from nearest node
        n_of_nodes += 1

        for id_ver, vertex in G.vertices.items():                           # iterate through all the vertices
            # id_ver = G.id_vertex[vertex]                     # get the id of the vertex
            if id_ver == id_new:
                continue
            distance_new_vert = calc_distance(q_new, vertex)    # calculate distance between new node and vertex node
            if distance_new_vert > radius:                      # if distance is greater than search radius - continue
                continue
            line = Line(vertex, q_new)                          # create Line object from new node to vertex
            if through_obstacle(line, obstacles):               # if the line goes through obstacle - continue
                continue
            if G.cost[id_new] > G.cost[id_ver] + distance_new_vert:  # if cost from new node to vertex is smaller
                G.cost[id_new] = G.cost[id_ver] + distance_new_vert  # new parent of vertex is id_new
                best_edge = (id_new, id_ver, distance_new_vert)      # than current cost, rewire the vertex to new

        G.parent[id_new] = best_edge[1]
        G.children[best_edge[1]].append(id_new)
        G.add_edge(*best_edge)

        # rewire
        for id_ver, vertex in G.vertices.items():  # iterate through all the vertices
            if id_ver == id_new:
                continue
            distance_new_vert = calc_distance(q_new, vertex)  # calculate distance between new node and vertex node
            if distance_new_vert > radius:  # if distance is greater than search radius - continue
                continue
            line = Line(vertex, q_new)  # create Line object from new node to vertex
            if through_obstacle(line, obstacles):  # if the line goes through obstacle - continue
                continue
            if G.cost[id_new] + distance_new_vert < G.cost[id_ver]:  # if cost from new node to vertex is smaller
                parent = G.parent[id_ver]
                del G.children[parent][G.children[parent].index(id_ver)]
                G.parent[id_ver] = id_new
                G.children[id_new].append(id_ver)
                G.cost[id_ver] = G.cost[id_new] + distance_new_vert
                G.edges[id_ver] = [(id_ver, id_new)]

        # delete random childless node if needed
        if n_of_nodes > max_nodes:
            childless_nodes = [node for node, children in G.children.items() if len(children) == 0]
            id_ver = random.choice(childless_nodes)
            G.remove_vertex(id_ver)
            n_of_nodes -= 1

        dist_to_goal = calc_distance(q_new, G.goal)                     # check if the goal has been reached
        if dist_to_goal < 2 * NODE_RADIUS:
            G.add_vertex(G.goal)
            G.add_edge(id_new, G.id_vertex[G.goal], dist_to_goal)
            print("********PATH FOUND**********")
            dijkstra2(G, q_new)
            break       # it should absolutely not stop here, but instead continue to grow the tree and search for optimal solution

        iter += 1
        # plot_graph(G, map.obstacles_c)
        # plt.show()

    return iter


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

    # G = Graph(start_node, goal_node, map_width, map_height)
    # iteration = RRT_star(G, iter_num=200, map=my_map, step_length=25, radius=30, bias=0)
    # print(f"RRT_star algorithm stopped at iteration number: {iteration}")
    # plot_graph(G, my_map.obstacles_c)
    # plt.show()
    #
    # G = Graph(start_node, goal_node, map_width, map_height)
    # iteration = RRT_star_FN(G, iter_num=500, map=my_map, step_length=25, radius=30, max_nodes=20, bias=0)
    # print(f"RRT_star_FN algorithm stopped at iteration number: {iteration}")
    # plot_graph(G, my_map.obstacles_c)
    #
    # plt.show()

    # RRT_STAR robi niepołączone z niczym gałęzie
    # jeden raz przeszedł node przez przeszkodę
    # jest problem z rewire w RRT_STAR