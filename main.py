# from collections import deque
# from collections import defaultdict
# import heapq as heap
import warnings

import matplotlib.pyplot as plt

from algorithm import RRT, RRT_star, RRT_star_FN
from graph import Graph
from map import Map

NODE_RADIUS = 5


warnings.filterwarnings("error")


def plot_graph(graph: Graph, obstacles: list):
    """
    Plot graph and obstacles.
    :param graph: Graph to plot
    :param obstacles: list of obstacles
    """
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

    plt.xlim(0, graph.width)
    plt.ylim(0, graph.height)

#
# def nearest_node(graph: Graph, vertex: tuple, obstacles: list):
#     """
#     Checks for the nearest node to the input node, check for crossing obstacles.
#      :param graph: Graph
#      :param vertex: position of the vertex
#      :param obstacles: list of obstacles
#      :return: new_vertex, new_id
#      """
#     try:
#         id = graph.id_vertex[vertex]
#         return vertex, id
#     except KeyError:
#         min_distance = float("inf")
#         new_id = None
#         new_vertex = None
#         for ver_id, ver in graph.vertices.items():      # was enumerate(graph.vertices)
#             line = Line(ver, vertex)
#             if through_obstacle(line, obstacles):
#                 continue
#             distance = calc_distance(ver, vertex)
#             if distance < min_distance:
#                 min_distance = distance
#                 new_id = ver_id
#                 new_vertex = ver
#
#         return new_vertex, new_id
#
#
# def new_node(to_vertex: tuple, from_vertex: tuple, max_length: float) -> tuple:
#     """
#     Return position of new node. from_vertex -> to_vertex, of given length.
#     :param to_vertex: position of vertex that marks the direction
#     :param from_vertex: position of parent vertex
#     :param max_length: maximum allowed length of the edge between two nodes
#     :return: position of new node
#     """
#     distance = calc_distance(to_vertex, from_vertex)
#     x_vect_norm = (to_vertex[0] - from_vertex[0]) / distance
#     y_vect_norm = (to_vertex[1] - from_vertex[1]) / distance
#     x_pos = from_vertex[0] + x_vect_norm * max_length
#     y_pos = from_vertex[1] + y_vect_norm * max_length
#     if distance > max_length:
#         return x_pos, y_pos
#     return to_vertex
#
#
# @time_function
# def RRT(G: Graph, iter_num: int, map: Map, step_length: float, bias: float = .0):
#     """
#     RRT algorithm.
#     :param G: Graph
#     :param iter_num: number of iterations for the algorithm
#     :param map: Map
#     :param step_length: maximum allowed length of the edge between two nodes
#     :param bias: 0-1, bias towards goal node
#     :return: number of iterations
#     """
#     pbar = tqdm(total=iter_num)
#     obstacles = map.obstacles_c
#     # goal_node = None
#     iter = 0
#     while iter < iter_num:
#         q_rand = G.random_node(bias=bias)                           # generate a new random node
#         if map.is_occupied_c(q_rand):                               # if new node's position is in an obstacle
#             continue
#         q_near, id_near = nearest_node(G, q_rand, obstacles)        # search for the nearest node
#         if q_near is None or q_rand == q_near:                                 # random node cannot be connected to nearest without obstacle
#             continue
#         q_new = new_node(q_rand, q_near, step_length)
#         id_new = G.add_vertex(q_new)
#         distance = calc_distance(q_new, q_near)
#         G.add_edge(id_new, id_near, distance)
#         # plot_graph(G, map.obstacles_c)
#         # plt.show()
#
#         if check_solution(G, q_new):
#             path, _ = find_path(G, id_new)
#             plot_path(G, path, "RRT")
#             # break
#
#         pbar.update(1)
#         iter += 1
#
#     pbar.close()
#     return iter
#
#
# @time_function
# def RRT_star(G, iter_num, map, step_length, radius, bias=.0):
#     """
#     RRT star algorithm.
#     :param G: Graph
#     :param iter_num: number of iterations for the algorithm
#     :param map: Map
#     :param step_length: maximum allowed length of the edge between two nodes
#     :param radius: radius of circular area that rewire algorithm will be performed on
#     :param bias: 0-1, bias towards goal node
#     :return: number of iterations
#     """
#     pbar = tqdm(total=iter_num)
#     obstacles = map.obstacles_c
#     best_edge = None
#     solution_found = False          # flag to know if the solution has already been found
#     best_path = {"path": [], "cost": float("inf")}                  # path with the smallest cost and its cost
#     finish_nodes_of_path = []        # ids of nodes that are the last nodes in found paths
#     iter = 0
#     while iter < iter_num:
#         q_rand = G.random_node(bias=bias)   # generate random node
#         if map.is_occupied_c(q_rand):       # if it's generated on an obstacle, continue
#             continue
#         q_near, id_near = nearest_node(G, q_rand, obstacles)    # find the nearest to the random node; change function to also include radius?
#         if q_near is None or q_rand == q_near:                                 # random node cannot be connected to nearest without obstacle
#             continue
#         q_new = new_node(q_rand, q_near, step_length)       # get position of the new node
#         id_new = G.add_vertex(q_new)                        # get id of the new node
#         cost_new_near = calc_distance(q_new, q_near)        # find cost from q_new to q_near
#         best_edge = (id_new, id_near, cost_new_near)
#         G.cost[id_new] = cost_new_near    # calculate cost for new node from nearest node
#         G.parent[id_new] = id_near
#
#         find_best_node(G, q_new, id_new, best_edge, radius, obstacles)
#         G.add_edge(*best_edge)
#
#         # rewire
#         rewire(G, q_new, id_new, radius, obstacles)
#
#         # check for solution
#         if check_solution(G, q_new):
#             path, _ = find_path(G, id_new)
#             plot_path(G, path, "RRT_STAR")
#             finish_nodes_of_path.append(id_new)
#             solution_found = True
#
#         # update cost of paths
#         for node in finish_nodes_of_path:
#             path, cost = find_path(G, node)
#             if cost < best_path["cost"]:
#                 best_path["path"] = path
#                 best_path["cost"] = cost
#
#         pbar.update(1)
#         iter += 1
#         # plt.pause(0.001)
#         # plt.clf()
#         # plot_graph(G, map.obstacles_c)
#         # if solution_found:
#         #     plot_path(G, best_path["path"], "RRT_STAR_FN", best_path["cost"])
#
#     pbar.close()
#     return iter
#
#
# @time_function
# def RRT_star_FN(G, iter_num, map, step_length, radius, max_nodes=200, bias=.0):
#     """
#     RRT star algorithm.
#     :param G: Graph
#     :param iter_num: number of iterations for the algorithm
#     :param map: Map
#     :param step_length: maximum allowed length of the edge between two nodes
#     :param radius: radius of circular area that rewire algorithm will be performed on
#     :param max_nodes: maximum number of nodes
#     :param bias: 0-1, bias towards goal node
#     :return: number of iterations
#     """
#     pbar = tqdm(total=iter_num)
#     obstacles = map.obstacles_c
#     best_edge = None
#     n_of_nodes = 1                  # only starting node at the beginning
#     solution_found = False          # flag to know if the solution has already been found
#     best_path = {"path": [], "cost": float("inf")}                  # path with the smallest cost and its cost
#     finish_nodes_of_path = []        # ids of nodes that are the last nodes in found paths
#     iter = 0
#     while iter < iter_num:
#         q_rand = G.random_node(bias=bias)               # generate random node
#         if map.is_occupied_c(q_rand):                   # if it's generated on an obstacle, continue
#             continue
#         q_near, id_near = nearest_node(G, q_rand,
#                                        obstacles)       # find the nearest to the random node; change function to also include radius?
#         if q_near is None or q_rand == q_near:          # random node cannot be connected to nearest without obstacle
#             continue
#         q_new = new_node(q_rand, q_near, step_length)   # get position of the new node
#         id_new = G.add_vertex(q_new)                    # get id of the new node
#         n_of_nodes += 1
#         cost_new_near = calc_distance(q_new, q_near)    # find cost from q_new to q_near
#         best_edge = (id_new, id_near, cost_new_near)
#         G.cost[id_new] = cost_new_near                  # calculate cost for new node from nearest node
#         G.parent[id_new] = id_near
#
#         find_best_node(G, q_new, id_new, best_edge, radius, obstacles)
#         G.add_edge(*best_edge)
#
#         # rewire
#         rewire(G, q_new, id_new, radius, obstacles)
#
#         # delete random childless node if needed
#         if n_of_nodes > max_nodes:
#             id_removed = delete_childless_node(G, id_new, best_path["path"])
#             if id_removed in finish_nodes_of_path:
#                 finish_nodes_of_path.remove(id_removed)
#             n_of_nodes -= 1
#
#         # check for solution
#         if check_solution(G, q_new):
#             path, _ = find_path(G, id_new)
#             plot_path(G, path, "RRT_STAR_FN")
#             # nodes_in_path += path
#             finish_nodes_of_path.append(id_new)
#             # current_path = path
#             solution_found = True
#             # break
#
#         # update cost of paths
#         for node in finish_nodes_of_path:
#             path, cost = find_path(G, node)
#             if cost < best_path["cost"]:
#                 best_path["path"] = path
#                 best_path["cost"] = cost
#
#         pbar.update(1)
#         iter += 1
#         # plt.pause(0.001)
#         # plt.clf()
#         # plot_graph(G, map.obstacles_c)
#         # if solution_found:
#         #     plot_path(G, best_path["path"], "RRT_STAR_FN", best_path["cost"])
#
#     pbar.close()
#     return iter
#
#
# def check_solution(G: Graph, q_new: tuple) -> bool:
#     """
#     Check if the solution has been found (node is close enough to goal node).
#     :param G: Graph
#     :param q_new: node to check
#     :return: True if solution found, False otherwise
#     """
#     dist_to_goal = calc_distance(q_new, G.goal)  # check if the goal has been reached
#     if dist_to_goal < 2 * NODE_RADIUS:
#         return True
#     return False
#
#
# def plot_path(G: Graph, path: list, title: str = "", cost: float = float("inf")):
#     """
#     Plot path.
#     :param G: Graph
#     :param path: list of ids of nodes that are in the path
#     :param title: title of the plot
#     :param cost: cost of the path
#     """
#     prev_node = G.goal
#     for point in path:
#         plt.plot((prev_node[0], G.vertices[point][0]), (prev_node[1], G.vertices[point][1]), c='red')
#         prev_node = G.vertices[point]
#     plt.title(title + f" cost: {round(cost, 2)}")
#
#
# def find_path(G: Graph, from_node: int) -> tuple:
#     """
#     Find path from from_node to start node.
#     :param G:
#     :param from_node:
#     :return: path, cost
#     """
#     path = []
#     node = from_node
#     cost = 0
#     while G.parent[node] is not None:
#         path.append(node)
#         cost += G.cost[node]
#         node = G.parent[node]
#     path.append(G.id_vertex[G.start])
#
#     return path, cost
#
#
# def delete_childless_node(G: Graph, id_new: int, path: list) -> int:
#     """
#     Delete random childless node from the graph.
#     :param G: Graph
#     :param id_new: id of node that won't be deleted
#     :param path: list of ids in the path
#     :return: id of node that has been deleted
#     """
#     childless_nodes = [node for node, children in G.children.items() if len(children) == 0] # and node != id_new
#     id_ver = random.choice(childless_nodes)
#     while id_ver in path or id_ver == id_new:
#         id_ver = random.choice(childless_nodes)
#     G.remove_vertex(id_ver)
#     return id_ver
#
#
# def find_best_node(G: Graph, q_new: tuple, id_new: int, best_edge: tuple, radius: float, obstacles: list) -> tuple:
#     """
#     Find a node that is optimal in terms of cost to the start node.
#     :param G: Graph
#     :param q_new: position of new node
#     :param id_new: id of new node
#     :param best_edge: best edge so far
#     :param radius: radius of search area
#     :param obstacles: list of obstacles
#     :return: id of best node
#     """
#     for id_ver, vertex in G.vertices.items():  # iterate through all the vertices
#         if id_ver == id_new: continue
#         distance_new_vert = calc_distance(q_new, vertex)  # calculate distance between new node and vertex node
#         if distance_new_vert > radius: continue  # if distance is greater than search radius - continue
#         line = Line(vertex, q_new)  # create Line object from new node to vertex
#         if through_obstacle(line, obstacles): continue  # if the line goes through obstacle - continue
#         # if G.cost[id_new] > G.cost[id_ver] + distance_new_vert:  # if cost from new node to vertex is smaller
#         if G.get_cost(id_new) > G.get_cost(id_ver) + distance_new_vert:
#             # G.cost[id_new] = G.cost[id_ver] + distance_new_vert  # than current cost, rewire the vertex to new
#             G.cost[id_new] = distance_new_vert
#             # best_edge = (id_new, id_ver, G.cost[id_ver] + distance_new_vert)
#             best_edge = (id_new, id_ver, distance_new_vert)
#
#     return best_edge
#
#
# def rewire(G: Graph, q_new: tuple, id_new: int, radius: float, obstacles: list):
#     """
#     Rewire procedure of the RRT_STAR algorithm.
#     :param G: Graph
#     :param q_new: position of new node
#     :param id_new: id of new node
#     :param radius: radius of search area
#     :param obstacles: list of obstacles
#     """
#     for id_ver, vertex in G.vertices.items():
#         if id_ver == G.id_vertex[G.start]: continue
#         if id_ver == id_new: continue
#         distance_new_vert = calc_distance(q_new, vertex)
#         if distance_new_vert > radius: continue
#         line = Line(vertex, q_new)
#         if through_obstacle(line, obstacles): continue
#         # if G.cost[id_new] + distance_new_vert < G.cost[id_ver]:
#         if G.get_cost(id_ver) > G.get_cost(id_new) + distance_new_vert:
#             parent = G.parent[id_ver]           # parent of the rewired node
#             del G.children[parent][G.children[parent].index(id_ver)]  # delete rewired node from it's parent children
#             G.parent[id_ver] = id_new           # set rewired node's parent to new node
#             G.children[id_new].append(id_ver)   # append rewired node to new node's children
#             # saved_cost = G.cost[id_ver] - (G.cost[id_new] + distance_new_vert)
#             # G.cost[id_ver] = G.cost[id_new] + distance_new_vert
#             G.cost[id_ver] = distance_new_vert
#             # update_cost(G, id_ver, saved_cost)


# def dijkstra(G, final_node):    # room for improvement with PriorityQueue or heapdict
#     id_start = G.id_vertex[G.start]
#     id_goal = G.id_vertex[G.goal]
#
#     nodes = list(G.neighbors.keys())
#     score = {node: float("inf") for node in nodes}
#     prev = {node: None for node in nodes}
#     score[id_goal] = 0
#     while nodes:
#         u = min(nodes, key=lambda node: score[node])
#         nodes.remove(u)
#         if score[u] == float("inf"):
#             break
#         for v, cost in G.neighbors[u]:
#             alt = score[u] + cost
#             if alt < score[v]:
#                 score[v] = alt
#                 prev[v] = u
#
#     path = deque()
#     u = id_start
#     while prev[u] is not None:
#         path.appendleft(G.vertices[u])
#         u = prev[u]
#     path.appendleft(G.vertices[u])
#     prev_node = G.vertices[id_goal]
#     for point in path:
#         plt.plot((prev_node[0], point[0]), (prev_node[1], point[1]), c='red')
#         prev_node = point
#     plt.title(f"Cost: {score[id_start]}")
#     return list(path)
#
#
# def dijkstra2(G, start_node):
#     closed = set()                                  # set of visited nodes
#     parent = {}                                     # dict of nodes parents
#     pq = []
#     node_scores = defaultdict(lambda: float("inf"))
#     node_scores[G.id_vertex[start_node]] = 0         # set score of start_node to 0
#     heap.heappush(pq, (0, G.id_vertex[start_node]))  # add first node to priority queue
#
#     while pq:
#         _, min_cost_node = heap.heappop(pq)         # get the node with lowest cost
#         if G.vertices[min_cost_node] == G.start:    # if this node is start node, end the loop, solution found
#             break
#         closed.add(min_cost_node)                   # add selected node to set of visited nodes
#
#         for v, cost in G.neighbors[min_cost_node]:  # iterate through each of the node's neighbors
#             if v in closed:                         # if the node is in set of visited nodes, continue
#                 continue
#             alt = node_scores[min_cost_node] + cost  # calculate alternative cost for the node
#             if alt < node_scores[v]:                # if the alternative cost is smaller than current
#                 parent[v] = min_cost_node           # set parent of this node to the min_node
#                 node_scores[v] = alt                # change the score of this node
#                 heap.heappush(pq, (alt, v))         # add this node to priority queue to check next
#
#     path = []
#     u = G.id_vertex[G.start]                        # add start vertex to path
#     while True:
#         try:
#             path.append(G.vertices[u])              # try to get parent of the node
#             u = parent[u]                           # go deeper into the tree
#         except KeyError:
#             break
#
#     prev_node = G.start
#     for point in path:
#         plt.plot((prev_node[0], point[0]), (prev_node[1], point[1]), c='red')
#         prev_node = point
#     plt.plot((prev_node[0], G.goal[0]), (prev_node[1], G.goal[1]), c='red')
#
#     return path, node_scores

if __name__ == '__main__':
    map_width = 300
    map_height = 300
    start_node = (50, 50)
    goal_node = (150, 90)
    my_map = Map((map_width, map_height), start_node, goal_node, NODE_RADIUS)
    my_map.generate_obstacles(obstacle_count=80, size=7)

    G = Graph(start_node, goal_node, map_width, map_height)
    iteration = RRT(G, iter_num=500, map=my_map, step_length=25, node_radius=NODE_RADIUS, bias=0)
    plot_graph(G, my_map.obstacles_c)
    print(f"RRT algorithm stopped at iteration number: {iteration}")
    plt.show()

    G = Graph(start_node, goal_node, map_width, map_height)
    iteration = RRT_star(G, iter_num=500, map=my_map, step_length=25, radius=30, node_radius=NODE_RADIUS, bias=0)
    print(f"RRT_star algorithm stopped at iteration number: {iteration}")
    plot_graph(G, my_map.obstacles_c)
    plt.show()

    G = Graph(start_node, goal_node, map_width, map_height)
    iteration = RRT_star_FN(G, iter_num=500, map=my_map, step_length=25, radius=30, node_radius=NODE_RADIUS, max_nodes=30, bias=0.02)
    print(f"RRT_star_FN algorithm stopped at iteration number: {iteration}")
    plot_graph(G, my_map.obstacles_c)

    # plt.figure()

    plt.ioff()
    plt.show()
