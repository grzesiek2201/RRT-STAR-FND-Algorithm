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
