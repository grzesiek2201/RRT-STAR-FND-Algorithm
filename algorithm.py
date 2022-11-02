from time import perf_counter
from functools import wraps
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random

from graph import Graph
from map import Map
from line import Line


def time_function(func):
    """ Wrapper for function timing
    :param func: function to time:"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = perf_counter()

        func(*args, **kwargs)

        end_time = perf_counter()
        total_time = round(end_time - start_time, 2)

        print(f"Execution time of {func.__name__}: {total_time}")

    return wrapper

@time_function
def RRT(G: Graph, iter_num: int, map: Map, step_length: float, node_radius: int, bias: float = .0):
    """
    RRT algorithm.
    :param G: Graph
    :param iter_num: number of iterations for the algorithm
    :param map: Map
    :param step_length: maximum allowed length of the edge between two nodes
    :param node_radius: radius of a node
    :param bias: 0-1, bias towards goal node
    :return: number of iterations
    """
    pbar = tqdm(total=iter_num)
    obstacles = map.obstacles_c
    # goal_node = None
    iter = 0
    while iter < iter_num:
        q_rand = G.random_node(bias=bias)                           # generate a new random node
        if map.is_occupied_c(q_rand):                               # if new node's position is in an obstacle
            continue
        q_near, id_near = nearest_node(G, q_rand, obstacles)        # search for the nearest node
        if q_near is None or q_rand == q_near:                                 # random node cannot be connected to nearest without obstacle
            continue
        q_new = new_node(q_rand, q_near, step_length)
        id_new = G.add_vertex(q_new)
        distance = calc_distance(q_new, q_near)
        G.add_edge(id_new, id_near, distance)
        # plot_graph(G, map.obstacles_c)
        # plt.show()

        if check_solution(G, q_new, node_radius):
            path, _ = find_path(G, id_new)
            plot_path(G, path, "RRT")
            # break

        pbar.update(1)
        iter += 1

    pbar.close()
    return iter


@time_function
def RRT_star(G, iter_num, map, step_length, radius, node_radius: int, bias=.0):
    """
    RRT star algorithm.
    :param G: Graph
    :param iter_num: number of iterations for the algorithm
    :param map: Map
    :param step_length: maximum allowed length of the edge between two nodes
    :param radius: radius of circular area that rewire algorithm will be performed on
    :param node_radius: radius of a node
    :param bias: 0-1, bias towards goal node
    :return: number of iterations
    """
    pbar = tqdm(total=iter_num)
    obstacles = map.obstacles_c
    best_edge = None
    solution_found = False          # flag to know if the solution has already been found
    best_path = {"path": [], "cost": float("inf")}                  # path with the smallest cost and its cost
    finish_nodes_of_path = []        # ids of nodes that are the last nodes in found paths
    iter = 0
    while iter < iter_num:
        q_rand = G.random_node(bias=bias)   # generate random node
        if map.is_occupied_c(q_rand):       # if it's generated on an obstacle, continue
            continue
        q_near, id_near = nearest_node(G, q_rand, obstacles)    # find the nearest to the random node; change function to also include radius?
        if q_near is None or q_rand == q_near:                                 # random node cannot be connected to nearest without obstacle
            continue
        q_new = new_node(q_rand, q_near, step_length)       # get position of the new node
        id_new = G.add_vertex(q_new)                        # get id of the new node
        cost_new_near = calc_distance(q_new, q_near)        # find cost from q_new to q_near
        best_edge = (id_new, id_near, cost_new_near)
        G.cost[id_new] = cost_new_near    # calculate cost for new node from nearest node
        G.parent[id_new] = id_near

        find_best_node(G, q_new, id_new, best_edge, radius, obstacles)
        G.add_edge(*best_edge)

        # rewire
        rewire(G, q_new, id_new, radius, obstacles)

        # check for solution
        if check_solution(G, q_new, node_radius):
            path, _ = find_path(G, id_new)
            plot_path(G, path, "RRT_STAR")
            finish_nodes_of_path.append(id_new)
            solution_found = True

        # update cost of paths
        for node in finish_nodes_of_path:
            path, cost = find_path(G, node)
            if cost < best_path["cost"]:
                best_path["path"] = path
                best_path["cost"] = cost

        pbar.update(1)
        iter += 1
        # plt.pause(0.001)
        # plt.clf()
        # plot_graph(G, map.obstacles_c)
        # if solution_found:
        #     plot_path(G, best_path["path"], "RRT_STAR_FN", best_path["cost"])

    pbar.close()
    return iter


@time_function
def RRT_star_FN(G, iter_num, map, step_length, radius, node_radius: int, max_nodes=200, bias=.0):
    """
    RRT star algorithm.
    :param G: Graph
    :param iter_num: number of iterations for the algorithm
    :param map: Map
    :param step_length: maximum allowed length of the edge between two nodes
    :param radius: radius of circular area that rewire algorithm will be performed on
    :param node_radius: radius of a node
    :param max_nodes: maximum number of nodes
    :param bias: 0-1, bias towards goal node
    :return: number of iterations
    """
    pbar = tqdm(total=iter_num)
    obstacles = map.obstacles_c
    best_edge = None
    n_of_nodes = 1                  # only starting node at the beginning
    solution_found = False          # flag to know if the solution has already been found
    best_path = {"path": [], "cost": float("inf")}                  # path with the smallest cost and its cost
    finish_nodes_of_path = []        # ids of nodes that are the last nodes in found paths
    iter = 0
    while iter < iter_num:
        q_rand = G.random_node(bias=bias)               # generate random node
        if map.is_occupied_c(q_rand):                   # if it's generated on an obstacle, continue
            continue
        q_near, id_near = nearest_node(G, q_rand,
                                       obstacles)       # find the nearest to the random node; change function to also include radius?
        if q_near is None or q_rand == q_near:          # random node cannot be connected to nearest without obstacle
            continue
        q_new = new_node(q_rand, q_near, step_length)   # get position of the new node
        id_new = G.add_vertex(q_new)                    # get id of the new node
        n_of_nodes += 1
        cost_new_near = calc_distance(q_new, q_near)    # find cost from q_new to q_near
        best_edge = (id_new, id_near, cost_new_near)
        G.cost[id_new] = cost_new_near                  # calculate cost for new node from nearest node
        G.parent[id_new] = id_near

        find_best_node(G, q_new, id_new, best_edge, radius, obstacles)
        G.add_edge(*best_edge)

        # rewire
        rewire(G, q_new, id_new, radius, obstacles)

        # delete random childless node if needed
        if n_of_nodes > max_nodes:
            id_removed = delete_childless_node(G, id_new, best_path["path"])
            if id_removed in finish_nodes_of_path:
                finish_nodes_of_path.remove(id_removed)
            n_of_nodes -= 1

        # check for solution
        if check_solution(G, q_new, node_radius):
            path, _ = find_path(G, id_new)
            plot_path(G, path, "RRT_STAR_FN")
            # nodes_in_path += path
            finish_nodes_of_path.append(id_new)
            # current_path = path
            solution_found = True
            # break

        # update cost of paths
        for node in finish_nodes_of_path:
            path, cost = find_path(G, node)
            if cost < best_path["cost"]:
                best_path["path"] = path
                best_path["cost"] = cost

        pbar.update(1)
        iter += 1
        # plt.pause(0.001)
        # plt.clf()
        # plot_graph(G, map.obstacles_c)
        # if solution_found:
        #     plot_path(G, best_path["path"], "RRT_STAR_FN", best_path["cost"])

    pbar.close()
    return iter


def intersection_circle(line: Line, circle: list) -> bool:
    """
    Check for intersection between a section and a circle.
    :param line: line
    :param circle: circle
    :return: True if intersects, False if doesn't intersect
    """
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


def through_obstacle(line: Line, obstacles: list) -> bool:  # only for circular obstacles for now
    """
    Check if line goes through obstacles.
    :param line: line to check
    :param obstacles: list of obstacles
    :return: True if collision, False if no collision
    """
    for obstacle in obstacles:
        if intersection_circle(line, obstacle):
            return True
    return False


def delta_solutions(delta: float, a: float, b: float) -> tuple:
    """
    Calculate solutions from delta and quadratic equation's coefficients.
    :param delta: delta
    :param a: coefficient of x^2
    :param b: coefficient of x
    :return: two solutions, x1 and x2
    """
    x1 = [(-b + delta**0.5) / (2 * a), None]
    x2 = [(-b - delta**0.5) / (2 * a), None]
    return x1, x2


def calc_delta(line: Line, circle: list) -> tuple:
    """
    Calculate delta of intersection between line and circle, as well as equation's coefficients and return the.m
    :param line: Line
    :param circle: circle
    :return: tuple of delta, coefficient of x^2, coefficient of x
    """
    x0 = circle[0][0]  # circle's center x coordinate
    y0 = circle[0][1]  # circle's center y coordinate
    r = circle[1]      # circle's radius
    a = (1 + line.dir**2)
    b = 2 * (-x0 + line.dir * line.const_term - line.dir * y0)
    c = -r ** 2 + x0 ** 2 + y0 ** 2 - 2 * y0 * line.const_term + line.const_term ** 2
    delta = b ** 2 - 4 * a * c
    return delta, a, b


def is_between(pb, p1, p2):
    """
    Check if pb lies on a section on (p1, p2).
    :param pb: point to check
    :param p1: first point on the line
    :param p2: second point on the line
    :return: True if pb is between p1 and p2, False otherwise
    """
    check1 = pb[0] > min(p1[0], p2[0])
    check2 = pb[0] < max(p1[0], p2[0])

    return check1 and check2


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


def nearest_node(graph: Graph, vertex: tuple, obstacles: list):
    """
    Checks for the nearest node to the input node, check for crossing obstacles.
     :param graph: Graph
     :param vertex: position of the vertex
     :param obstacles: list of obstacles
     :return: new_vertex, new_id
     """
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


def new_node(to_vertex: tuple, from_vertex: tuple, max_length: float) -> tuple:
    """
    Return position of new node. from_vertex -> to_vertex, of given length.
    :param to_vertex: position of vertex that marks the direction
    :param from_vertex: position of parent vertex
    :param max_length: maximum allowed length of the edge between two nodes
    :return: position of new node
    """
    distance = calc_distance(to_vertex, from_vertex)
    x_vect_norm = (to_vertex[0] - from_vertex[0]) / distance
    y_vect_norm = (to_vertex[1] - from_vertex[1]) / distance
    x_pos = from_vertex[0] + x_vect_norm * max_length
    y_pos = from_vertex[1] + y_vect_norm * max_length
    if distance > max_length:
        return x_pos, y_pos
    return to_vertex


def check_solution(G: Graph, q_new: tuple, node_radius: int) -> bool:
    """
    Check if the solution has been found (node is close enough to goal node).
    :param G: Graph
    :param q_new: node to check
    :param node_radius: radius of node
    :return: True if solution found, False otherwise
    """
    dist_to_goal = calc_distance(q_new, G.goal)  # check if the goal has been reached
    if dist_to_goal < 2 * node_radius:
        return True
    return False


def plot_path(G: Graph, path: list, title: str = "", cost: float = float("inf")):
    """
    Plot path.
    :param G: Graph
    :param path: list of ids of nodes that are in the path
    :param title: title of the plot
    :param cost: cost of the path
    """
    prev_node = G.goal
    for point in path:
        plt.plot((prev_node[0], G.vertices[point][0]), (prev_node[1], G.vertices[point][1]), c='red')
        prev_node = G.vertices[point]
    plt.title(title + f" cost: {round(cost, 2)}")


def find_path(G: Graph, from_node: int) -> tuple:
    """
    Find path from from_node to start node.
    :param G:
    :param from_node:
    :return: path, cost
    """
    path = []
    node = from_node
    cost = 0
    while G.parent[node] is not None:
        path.append(node)
        cost += G.cost[node]
        node = G.parent[node]
    path.append(G.id_vertex[G.start])

    return path, cost


def delete_childless_node(G: Graph, id_new: int, path: list) -> int:
    """
    Delete random childless node from the graph.
    :param G: Graph
    :param id_new: id of node that won't be deleted
    :param path: list of ids in the path
    :return: id of node that has been deleted
    """
    childless_nodes = [node for node, children in G.children.items() if len(children) == 0] # and node != id_new
    id_ver = random.choice(childless_nodes)
    while id_ver in path or id_ver == id_new:
        id_ver = random.choice(childless_nodes)
    G.remove_vertex(id_ver)
    return id_ver


def find_best_node(G: Graph, q_new: tuple, id_new: int, best_edge: tuple, radius: float, obstacles: list) -> tuple:
    """
    Find a node that is optimal in terms of cost to the start node.
    :param G: Graph
    :param q_new: position of new node
    :param id_new: id of new node
    :param best_edge: best edge so far
    :param radius: radius of search area
    :param obstacles: list of obstacles
    :return: id of best node
    """
    for id_ver, vertex in G.vertices.items():  # iterate through all the vertices
        if id_ver == id_new: continue
        distance_new_vert = calc_distance(q_new, vertex)  # calculate distance between new node and vertex node
        if distance_new_vert > radius: continue  # if distance is greater than search radius - continue
        line = Line(vertex, q_new)  # create Line object from new node to vertex
        if through_obstacle(line, obstacles): continue  # if the line goes through obstacle - continue
        # if G.cost[id_new] > G.cost[id_ver] + distance_new_vert:  # if cost from new node to vertex is smaller
        if G.get_cost(id_new) > G.get_cost(id_ver) + distance_new_vert:
            # G.cost[id_new] = G.cost[id_ver] + distance_new_vert  # than current cost, rewire the vertex to new
            G.cost[id_new] = distance_new_vert
            # best_edge = (id_new, id_ver, G.cost[id_ver] + distance_new_vert)
            best_edge = (id_new, id_ver, distance_new_vert)

    return best_edge


def rewire(G: Graph, q_new: tuple, id_new: int, radius: float, obstacles: list):
    """
    Rewire procedure of the RRT_STAR algorithm.
    :param G: Graph
    :param q_new: position of new node
    :param id_new: id of new node
    :param radius: radius of search area
    :param obstacles: list of obstacles
    """
    for id_ver, vertex in G.vertices.items():
        if id_ver == G.id_vertex[G.start]: continue
        if id_ver == id_new: continue
        distance_new_vert = calc_distance(q_new, vertex)
        if distance_new_vert > radius: continue
        line = Line(vertex, q_new)
        if through_obstacle(line, obstacles): continue
        # if G.cost[id_new] + distance_new_vert < G.cost[id_ver]:
        if G.get_cost(id_ver) > G.get_cost(id_new) + distance_new_vert:
            parent = G.parent[id_ver]           # parent of the rewired node
            del G.children[parent][G.children[parent].index(id_ver)]  # delete rewired node from it's parent children
            G.parent[id_ver] = id_new           # set rewired node's parent to new node
            G.children[id_new].append(id_ver)   # append rewired node to new node's children
            # saved_cost = G.cost[id_ver] - (G.cost[id_new] + distance_new_vert)
            # G.cost[id_ver] = G.cost[id_new] + distance_new_vert
            G.cost[id_ver] = distance_new_vert
            # update_cost(G, id_ver, saved_cost)


def calc_distance(p1: tuple, p2: tuple) -> float:
    """
    Calculate distance between two points.
    :param p1: point 1
    :param p2: point 2
    :return: distance between points
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))
