from time import perf_counter
from functools import wraps
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random

from graph import Graph
from map import Map
from line import Line

from scipy.spatial import KDTree, cKDTree


def time_function(func):
    """
    Wrapper for function timing
    :param func: function to time
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = perf_counter()

        args = func(*args, **kwargs)

        end_time = perf_counter()
        total_time = round(end_time - start_time, 2)

        print(f"Execution time of {func.__name__}: {total_time}")
        return args

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

    iter = 0
    while iter < iter_num:
        q_rand = G.random_node(bias=bias)  # generate a new random node
        if map.is_occupied_c(q_rand):  # if new node's position is in an obstacle
            continue

        potential_vertices_list = list(G.vertices.values())
        kdtree = cKDTree(np.array(potential_vertices_list))  # create KDtree (2Dtree) and pass it to nearest_node function
        q_near, id_near = nearest_node_kdtree(G, q_rand, obstacles, kdtree=kdtree)
        if q_near is None or (q_rand == q_near).all():  # random node cannot be connected to nearest without obstacle
            continue
        # q_near, id_near = nearest_node(G, q_rand, obstacles)  # search for the nearest node
        # if q_near is None or (q_rand == q_near):  # random node cannot be connected to nearest without obstacle
        #     continue
        q_new = steer(q_rand, q_near, step_length)
        id_new = G.add_vertex(q_new)
        distance = calc_distance(q_new, q_near)
        G.add_edge(id_new, id_near, distance)

        if check_solution(G, q_new, node_radius):
            path, cost = find_path(G, id_new, G.id_vertex[G.start])
            plot_path(G, path, "RRT", cost)
            break

        pbar.update(1)
        iter += 1

        # plt.pause(0.001)
        # plt.clf()
        # plot_graph(G, map.obstacles_c)
        # plt.xlim((-200, 200))
        # plt.ylim((-200, 200))

    pbar.close()
    return iter


@time_function
def RRT_star(G, iter_num, map, step_length, radius, node_radius: int, bias=.0, live_update=False) -> tuple:
    """
    RRT star algorithm.
    :param G: Graph
    :param iter_num: number of iterations for the algorithm
    :param map: Map
    :param step_length: maximum allowed length of the edge between two nodes
    :param radius: radius of circular area that rewire algorithm will be performed on
    :param node_radius: radius of a node
    :param bias: 0-1, bias towards goal node
    :param live_update: live presentation of the algorithm on plot
    :return: number of iterations, the best path with cost
    """
    pbar = tqdm(total=iter_num)
    obstacles = map.obstacles_c
    best_edge = None
    solution_found = False  # flag to know if the solution has already been found
    best_path = {"path": [], "cost": float("inf")}  # path with the smallest cost and its cost
    finish_nodes_of_path = []  # ids of nodes that are the last nodes in found paths
    iter = 0
    while iter < iter_num:
        q_rand = G.random_node(bias=bias)  # generate random node
        if map.is_occupied_c(q_rand):  # if it's generated on an obstacle, continue
            continue

        potential_vertices_list = list(G.vertices.values())
        kdtree = cKDTree(
            np.array(potential_vertices_list))  # create KDtree (2Dtree) and pass it to nearest_node function
        q_near, id_near = nearest_node_kdtree(G, q_rand, obstacles, kdtree=kdtree)
        if q_near is None or (q_rand == q_near).all():  # random node cannot be connected to nearest without obstacle
            continue
        # q_near, id_near = nearest_node(G, q_rand,
        #                                obstacles)  # find the nearest to the random node; change function to also include radius?
        # if q_near is None or q_rand == q_near:  # random node cannot be connected to nearest without obstacle
        #     continue

        q_new = steer(q_rand, q_near, step_length)  # get position of the new node
        if map.is_occupied_c(q_new): continue
        id_new = G.add_vertex(q_new)  # get id of the new node
        cost_new_near = calc_distance(q_new, q_near)  # find cost from q_new to q_near
        best_edge = (id_new, id_near, cost_new_near)
        G.cost[id_new] = cost_new_near  # calculate cost for new node from nearest node
        G.parent[id_new] = id_near

        # KDTree performs better than brute-force
        kdtree = KDTree(list(G.vertices.values()))
        choose_parent_kdtree(G, q_new, id_new, best_edge, radius, obstacles, kdtree=kdtree)
        # choose_parent(G, q_new, id_new, best_edge, radius, obstacles)
        G.add_edge(*best_edge)


        # create a new 2Dtree? not sure
        # rewire
        rewire(G, q_new, id_new, radius, obstacles)

        # check for solution
        if check_solution(G, q_new, node_radius):
            path, cost = find_path(G, id_new, G.id_vertex[G.start])
            finish_nodes_of_path.append(id_new)
            solution_found = True

            best_path["path"] = path
            best_path["cost"] = cost
            # plot_path(G, path, "RRT_STAR", cost)
            # break

        # update cost of paths
        for node in finish_nodes_of_path:
            path, cost = find_path(G, node, G.id_vertex[G.start])
            if cost < best_path["cost"]:
                best_path["path"] = path
                best_path["cost"] = cost

        pbar.update(1)
        iter += 1
        if live_update:
            plt.pause(0.001)
            plt.clf()
            plot_graph(G, map.obstacles_c)
            if solution_found:
                plot_path(G, best_path["path"], "RRT_STAR", best_path["cost"])

    if solution_found:
        plot_path(G, best_path["path"], "RRT_STAR", best_path["cost"])

    pbar.close()

    return iter, best_path


@time_function
def RRT_star_FN(G, iter_num, map, step_length, radius, node_radius: int, max_nodes=200, bias=.0,
                live_update: bool = False):
    """
    RRT star FN algorithm.
    :param G: Graph
    :param iter_num: number of iterations for the algorithm
    :param map: Map
    :param step_length: maximum allowed length of the edge between two nodes
    :param radius: radius of circular area that rewire algorithm will be performed on
    :param node_radius: radius of a node
    :param max_nodes: maximum number of nodes
    :param bias: 0-1, bias towards goal node
    :param live_update: live presentation of the algorithm on plot
    :return: number of iterations
    """
    # plot_graph(G, map.obstacles_c)
    # plt.pause(3)

    pbar = tqdm(total=iter_num)
    obstacles = map.obstacles_c
    best_edge = None
    n_of_nodes = 1  # only starting node at the beginning
    solution_found = False  # flag to know if the solution has already been found
    best_path = {"path": [], "cost": float("inf")}  # path with the smallest cost and its cost
    finish_nodes_of_path = []  # ids of nodes that are the last nodes in found paths
    iter = 0
    while iter < iter_num:
        q_rand = G.random_node(bias=bias)  # generate random node
        if map.is_occupied_c(q_rand):  # if it's generated on an obstacle, continue
            continue

        potential_vertices_list = list(G.vertices.values())
        kdtree = cKDTree(
            np.array(potential_vertices_list))  # create KDtree (2Dtree) and pass it to nearest_node function
        q_near, id_near = nearest_node_kdtree(G, q_rand, obstacles, kdtree=kdtree)
        if q_near is None or (q_rand == q_near).all():  # random node cannot be connected to nearest without obstacle
            continue
        # q_near, id_near = nearest_node(G, q_rand,
        #                                obstacles)  # find the nearest to the random node; change function to also include radius?
        # if q_near is None or q_rand == q_near:  # random node cannot be connected to nearest without obstacle
        #     continue

        q_new = steer(q_rand, q_near, step_length)  # get position of the new node
        if map.is_occupied_c(q_new): continue
        id_new = G.add_vertex(q_new)  # get id of the new node
        n_of_nodes += 1
        cost_new_near = calc_distance(q_new, q_near)  # find cost from q_new to q_near
        best_edge = (id_new, id_near, cost_new_near)
        G.cost[id_new] = cost_new_near  # calculate cost for new node from nearest node
        G.parent[id_new] = id_near

        # KDTree performs worse than brute-force
        kdtree = KDTree(list(G.vertices.values()))
        choose_parent_kdtree(G, q_new, id_new, best_edge, radius, obstacles, kdtree=kdtree)
        # choose_parent(G, q_new, id_new, best_edge, radius, obstacles)
        G.add_edge(*best_edge)

        # rewire
        rewire(G, q_new, id_new, radius, obstacles)

        # delete random childless node if needed
        if n_of_nodes > max_nodes:
            id_removed = forced_removal(G, id_new, best_path["path"])
            if id_removed in finish_nodes_of_path:
                finish_nodes_of_path.remove(id_removed)
            n_of_nodes -= 1

        # check for solution
        if check_solution(G, q_new, node_radius):
            path, _ = find_path(G, id_new, G.id_vertex[G.start])
            finish_nodes_of_path.append(id_new)
            solution_found = True
            # break

        # update cost of paths
        for node in finish_nodes_of_path:
            path, cost = find_path(G, node, G.id_vertex[G.start])
            if cost < best_path["cost"]:
                best_path["path"] = path
                best_path["cost"] = cost

        pbar.update(1)
        iter += 1
        if live_update:
            plt.pause(0.001)
            plt.clf()
            plot_graph(G, map.obstacles_c)
            if solution_found:
                plot_path(G, best_path["path"], "RRT_STAR_FN", best_path["cost"])

    if solution_found:
        plot_path(G, best_path["path"], "RRT_STAR_FN", best_path["cost"])
    pbar.close()
    return iter, best_path


def RRT_STAR_FND(G: Graph, iter_num: int, map: Map, step_length: int, radius: float,
                 node_radius: int, max_nodes: int = 200, bias: float = .0, live_update: bool = False) -> None:
    """
    RRT star FN Dynamic algorithm.
    :param G: Graph
    :param iter_num: number of iterations for the algorithm
    :param map: Map
    :param step_length: maximum allowed length of the edge between two nodes
    :param radius: radius of circular area that rewire algorithm will be performed on
    :param node_radius: radius of a node
    :param max_nodes: maximum number of nodes
    :param bias: 0-1, bias towards goal node
    :param live_update: live presentation of the algorithm on plot
    :return: number of iterations
    """
    iter, best_path = RRT_star_FN(G=G, iter_num=iter_num, map=map, step_length=step_length,
                                  radius=radius, node_radius=node_radius, max_nodes=max_nodes, bias=bias,
                                  live_update=live_update)
    p_current = G.start
    init_movement()


def init_movement():
    pass


def select_branch(G: Graph, current_node: int, path: list, map: Map) -> list:
    """
    Select branch algorithm used in RRT_*_FND. Deletes all the nodes that are no longer in path and their children.
    :param G: Graph
    :param current_node: node that has been reached and parents of which and their children will be deleted
    :param path: previous path
    :param map: Map
    :return: new path
    """
    G.start = G.vertices[current_node]
    parent = G.parent[current_node]
    G.parent[current_node] = None  # set parent of current node to None
    del G.children[parent][
        G.children[parent].index(current_node)]  # delete current node from previous parent's children

    plt.figure()
    plot_graph(G, map.obstacles_c)
    plt.show()

    new_path = path[: path.index(current_node) + 1]
    stranded_nodes = path[path.index(current_node) + 1:]
    for stranded_node in reversed(stranded_nodes):
        if stranded_node == current_node: break
        remove_children(G, stranded_node, path)
        plt.figure()
        plot_graph(G, map.obstacles_c)
        plt.show()

    for node in stranded_nodes:
        G.remove_vertex(node)

    return new_path


def remove_children(G: Graph, id_node: int, path: list) -> None:
    """
    Removes all children of specified node if they are not in path.
    :param G: Graph
    :param id_node: id of node that children will be deleted
    :param path: current path
    :return: None
    """
    nodes_children = G.children[id_node].copy()
    for id_child in nodes_children:
        if id_child in path: continue
        if len(G.children[id_child]) != 0:
            remove_children(G, id_child, path)
        G.remove_vertex(id_child)


def valid_path(G: Graph, path: list, map: Map, previous_root: int) -> int:
    """
    ValidPath algorithm that deletes nodes colliding with obstacles and these nodes' children
    :param G: Graph
    :param path: previous path
    :param map: Map
    :param previous_root: id of previous tree's root node
    :return: id of node that is the root of the new, separated tree
    """
    id_separate_root = previous_root
    nodes_in_path = [(id_node, G.vertices[id_node]) for id_node in path]
    for id_node, pos_node in reversed(nodes_in_path):
        if map.is_occupied_c(pos_node):  # if node that is in path collides with an obstacle, delete it
            remove_children(G, id_node, path)
            id_remaining_child = G.children[id_node][0]  # the only remaining children of this node is in path
            G.remove_vertex(id_node)  # remove node that collides with an obstacle
            G.parent[id_remaining_child] = None  # set the separate root node's parent to None
            id_separate_root = id_remaining_child

            plt.pause(0.001)
            plt.clf()
            plot_graph(G, map.obstacles_c)

    return id_separate_root


def reconnect(G: Graph, path: list, map: Map, step_size: float, id_root: int, id_separate_root: int,
              last_node: int) -> tuple:
    """

    :param G: Graph
    :param path: current known path (although not leading to goal because SearchBranch and ValidPath have been executed)
    :param map: Map
    :param step_size: step size
    :param id_root: root of the original tree
    :param id_separate_root: root of the separated tree
    :param last_node: last node of the original path
    :return: tuple with path and cost
    """
    reconnected = False
    close_root_nodes = get_near_nodes(G, id_separate_root, step_size, path[-1])  # nodes from root tree that are close
    pos_separate = G.vertices[id_separate_root]
    for node in close_root_nodes:
        line = Line(G.vertices[node], pos_separate)
        if not through_obstacle(line, map.obstacles_c):
            cost = calc_distance(G.vertices[node], pos_separate)
            G.add_edge(id_separate_root, node, cost)
            reconnected = True
            break

    plt.figure()
    plot_graph(G, map.obstacles_c)
    plt.show()

    path_and_cost = None
    if reconnected:
        path_and_cost = find_path(G, last_node, id_root)
        plot_graph(G, map.obstacles_c)
        plot_path(G, path_and_cost[0], "after RECONNECT", path_and_cost[1])

    return path_and_cost


def check_for_tree_associativity(G: Graph, root_node: int, node_to_check: int) -> bool:
    """
    Check if node belongs to a tree with root at root_node
    :param G: Graph
    :param root_node: root of the tree
    :param node_to_check: node that will be checked
    :return: True if node_to_check belongs to tree
    """
    node = node_to_check
    parent = G.parent[node]
    while parent is not None:
        node = G.parent[node]
        parent = G.parent[node]
    return node == root_node


def get_near_nodes(G: Graph, node_to_check: int, radius: float, root_node: int) -> list:
    """
    Get list of ids of nodes that are within radius from node_to_check.
    :param G: Graph
    :param node_to_check: node to check
    :param radius: radius of search area
    :return: list of ids of nearby nodes
    """
    distances = get_distance_dict(G, node_to_check)
    id_near_nodes = [id_ver for id_ver, cost in distances.items() if
                     cost <= radius and check_for_tree_associativity(G, root_node, id_ver)]
    # id_near_nodes = []
    # for id_ver, cost in distances.items():
    #     if cost <= radius:
    #         if check_for_tree_associativity(G, root_node, id_ver):
    #             id_near_nodes.append(id_ver)
    return id_near_nodes


def test_select_branch():
    map_width = 200
    map_height = 200
    start = (50, 50)
    goal = (150, 150)
    NODE_RADIUS = 5
    step_length = 15
    my_map = Map((map_width, map_height), start, goal, NODE_RADIUS)
    my_map.generate_obstacles(obstacle_count=45, size=7)
    G = Graph(start, goal, map_width, map_height)
    # iteration, best_path = RRT_star(G, iter_num=500, map=my_map, step_length=step_length, radius=15, node_radius=NODE_RADIUS, bias=0)
    iteration, best_path = RRT_star_FN(G, iter_num=500, map=my_map, step_length=step_length, radius=15,
                                       node_radius=NODE_RADIUS, max_nodes=30, bias=0)

    plot_graph(G, my_map.obstacles_c)
    plt.pause(0.001)

    last_node = best_path["path"][0]

    id_to_remove = list(reversed(best_path["path"]))[4]

    best_path["path"] = select_branch(G, id_to_remove, best_path["path"], my_map)

    my_map.add_obstacles([[G.vertices[best_path["path"][6]], 7]])
    my_map.add_obstacles([[G.vertices[best_path["path"][7]], 7]])

    plt.figure()
    plot_graph(G, my_map.obstacles_c)
    plt.show()

    id_separate_root = valid_path(G, best_path["path"], my_map, id_to_remove)

    print(f"RRT_star algorithm stopped at iteration number: {iteration}")
    plt.figure()
    plot_graph(G, my_map.obstacles_c)
    plt.show()

    root_node = G.id_vertex[G.start]
    # check_for_tree_associativity(G, id_separate_root, G.children[G.children[id_separate_root][0]][0])
    ret_value = reconnect(G, best_path["path"], my_map, step_length*5, root_node, id_separate_root, last_node)
    if ret_value is None:
        regrow(G=G, map=my_map, step_length=step_length, radius=step_length, id_root=root_node,
               id_separate_root=id_separate_root, last_node=last_node, bias=0.02)


def regrow(G: Graph, map: Map, step_length: float, radius: float, id_root: int,
           id_separate_root: int, last_node: int, bias: float):
    separate_tree = [node for node in G.vertices if check_for_tree_associativity(G, id_separate_root, node)]

    iter_num = 500
    iter = 0
    reconnected = False
    n_of_nodes = len(G.vertices)
    obstacles = map.obstacles_c
    while iter < iter_num and not reconnected:
        q_rand = G.random_node(bias=bias)  # generate random node
        if map.is_occupied_c(q_rand): continue  # if it's generated on an obstacle, continue
        q_near, id_near = nearest_node(G, q_rand, obstacles, separate_tree)  # find the nearest to the random node
        if q_near is None or q_rand == q_near: continue  # random node cannot be connected to nearest without obstacle
        q_new = steer(q_rand, q_near, step_length)  # get position of the new node
        if map.is_occupied_c(q_new): continue
        id_new = G.add_vertex(q_new)  # get id of the new node
        n_of_nodes += 1
        cost_new_near = calc_distance(q_new, q_near)  # find cost from q_new to q_near
        best_edge = (id_new, id_near, cost_new_near)
        G.cost[id_new] = cost_new_near  # calculate cost for new node from nearest node
        G.parent[id_new] = id_near

        choose_parent(G, q_new, id_new, best_edge, radius, obstacles, separate_tree)
        G.add_edge(*best_edge)

        iter += 1

        for node in separate_tree:  # try to reconnect root tree with separate tree
            line = Line(q_new, G.vertices[node])
            if through_obstacle(line, obstacles): continue
            if calc_distance(q_new, G.vertices[node]) < step_length:
                # delete_ancestors(G, node)
                reconnect_ancestors(G, node)
                G.add_edge(node, id_new, calc_distance(q_new, G.vertices[node]))
                reconnected = True
                break

        plt.pause(0.001)
        plt.clf()
        plot_graph(G, obstacles)

    path = find_path(G, last_node, id_root)
    plt.figure()
    plot_graph(G, obstacles)
    plot_path(G, path[0], "after regrow")
    plt.show()


def reconnect_ancestors(G: Graph, id_node: int) -> None:
    node = id_node
    parent = G.parent[node]
    if parent is None:
        return
    while G.parent[parent] is not None:
        parent_parent = G.parent[parent]
        G.parent[parent] = node
        G.children[node].append(parent)
        del G.children[parent][G.children[parent].index(node)]
        node = parent
        parent = parent_parent
    G.parent[parent] = node


def delete_ancestors(G: Graph, id_node: int) -> None:
    node = id_node
    parent = G.parent[node]
    while parent is not None:
        node = parent
        remove_children(G, node, [])
        parent = G.parent[node]


def intersection_circle(line: Line, circle: list) -> bool:
    """
    Check for intersection between a section and a circle.
    :param line: line
    :param circle: circle
    :return: True if intersects, False if doesn't intersect
    """
    p1, p2 = line.p1, line.p2
    if line.dir == float("inf"):  # when the line is vertical, thus calculating delta doesn't work
        if abs(circle[0][0] - line.x_pos) <= circle[1]:
            return True
        else:
            return False
    delta, a, b = calc_delta(line, circle)
    if delta < 0:  # no real solutions
        return False
    ip1, ip2 = delta_solutions(delta, a, b)  # intersection point 1's x; intersection point 2's x
    ip1[1] = line.calculate_y(ip1[0])
    ip2[1] = line.calculate_y(ip2[0])
    # if (ip1[0] < 0 and ip2[0] < 0) or (ip1[1] < 0 and ip2[1] < 0):  # outside the map
    #     return False
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
    x1 = [(-b + delta ** 0.5) / (2 * a), None]
    x2 = [(-b - delta ** 0.5) / (2 * a), None]
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
    r = circle[1]  # circle's radius
    a = (1 + line.dir ** 2)
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

    plt.scatter(xes, yes, c='gray')  # plotting nodes
    plt.scatter(graph.start[0], graph.start[1], c='#49ab1f', s=50)
    plt.scatter(graph.goal[0], graph.goal[1], c='red', s=50)

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


def nearest_node_kdtree(G: Graph, vertex: tuple, obstacles: list, separate_tree_nodes: list = (), kdtree=None):
    """
    Checks for the nearest node to the input node, check for crossing obstacles.
     :param G: Graph
     :param vertex: position of the vertex which neighbors are being sought for
     :param obstacles: list of obstacles
     :param separate_tree_nodes: list of nodes that are in the separate tree
     :return: new_vertex, new_id
     """
    try:
        id = G.id_vertex[vertex]
        return np.array(vertex), id
    except KeyError:
        closest_id = None
        closest_pos = None
        nn = 1
        while True:
            d, i = kdtree.query(vertex, k=nn)
            if nn == 1:
                closest_pos = kdtree.data[i]
            else:
                closest_pos = kdtree.data[i[-1]]
            closest_id = G.id_vertex[closest_pos[0], closest_pos[1]]
            line = Line(vertex, closest_pos)
            nn += 1
            if not through_obstacle(line, obstacles):
                break
            elif nn > len(G.vertices):
                closest_pos = np.array(vertex)
                closest_id = None
                break

        return closest_pos, closest_id


def nearest_node(G: Graph, vertex: tuple, obstacles: list, separate_tree_nodes: list = ()):
    """
    Checks for the nearest node to the input node, check for crossing obstacles.
     :param G: Graph
     :param vertex: position of the vertex which neighbors are being sought for
     :param obstacles: list of obstacles
     :param separate_tree_nodes: list of nodes that are in the separate tree
     :return: new_vertex, new_id
     """
    try:
        id = G.id_vertex[vertex]
        return vertex, id
    except KeyError:
        min_distance = float("inf")
        new_id = None
        new_vertex = None
        for ver_id, ver in G.vertices.items():
            if ver_id in separate_tree_nodes: continue
            line = Line(ver, vertex)
            if through_obstacle(line, obstacles): continue
            distance = calc_distance(ver, vertex)
            if distance < min_distance:
                min_distance = distance
                new_id = ver_id
                new_vertex = ver

        return new_vertex, new_id


def steer(to_vertex: tuple, from_vertex: tuple, max_length: float) -> tuple:
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
        plt.plot((prev_node[0], G.vertices[point][0]), (prev_node[1], G.vertices[point][1]), c='#057af7', linewidth=2)
        prev_node = G.vertices[point]
    plt.title(title + f" cost: {round(cost, 2)}")


def find_path(G: Graph, from_node: int, root_node: int) -> tuple:
    """
    Find path from from_node to start node.
    :param G:
    :param from_node:
    :return: path, cost
    """
    path = []
    node = from_node
    cost = 0
    try:
        while node != root_node:
            path.append(node)
            cost += G.cost[node]
            node = G.parent[node]
        path.append(root_node)
    except Exception:
        pass

    return path, cost


def forced_removal(G: Graph, id_new: int, path: list) -> int:
    """
    Delete random childless node from the graph.
    :param G: Graph
    :param id_new: id of node that won't be deleted
    :param path: list of ids of the nodes in path
    :return: id of node that has been deleted
    """
    id_last_in_path = -1
    if path:
        id_last_in_path = path[0]

    childless_nodes = [node for node, children in G.children.items() if len(children) == 0]  # and node != id_new
    id_ver = random.choice(childless_nodes)

    while id_ver == id_new or id_ver == id_last_in_path:
        id_ver = random.choice(childless_nodes)
    G.remove_vertex(id_ver)

    return id_ver


def choose_parent_kdtree(G: Graph, q_new: tuple, id_new: int, best_edge: tuple,
                  radius: float, obstacles: list, separate_tree_nodes: list = (), kdtree=None) -> tuple:
    """
    Find a node that is optimal in terms of cost to the start node.
    :param G: Graph
    :param q_new: position of new node
    :param id_new: id of new node
    :param best_edge: best edge so far
    :param radius: radius of search area
    :param obstacles: list of obstacles
    :param separate_tree_nodes: list if ids of nodes that are in the separate tree
    :return: id of best node
    """

    i = kdtree.query_ball_point(q_new, r=radius)
    # calc_distances between new node and nodes in radius

    in_radius_pos = kdtree.data[i]                                              # pos of point in radius
    in_radius_ids = [G.id_vertex[pos[0], pos[1]] for pos in in_radius_pos]      # id of points in radius

    costs = np.linalg.norm(in_radius_pos - q_new, axis=1)
    # new_costs = [G.get_cost(id_in_radius) + costs[] for id_in_radius in in_radius_ids]

    for id_ver, vertex, cost in zip(in_radius_ids, in_radius_pos, costs):
        line = Line(vertex, q_new)
        if through_obstacle(line, obstacles): continue
        if G.get_cost(id_new) > G.get_cost(id_ver) + cost:
            G.cost[id_new] = cost
            best_edge = (id_new, id_ver, cost)

    # for id_ver, vertex in G.vertices.items():  # iterate through all the vertices
    #     if id_ver == id_new: continue
    #     distance_new_vert = calc_distance(q_new, vertex)  # calculate distance between new node and vertex node
    #     if distance_new_vert > radius: continue  # if distance is greater than search radius - continue
    #     line = Line(vertex, q_new)  # create Line object from new node to vertex
    #     if through_obstacle(line, obstacles): continue  # if the line goes through obstacle - continue
    #     if G.get_cost(id_new) > G.get_cost(id_ver) + distance_new_vert:  # if cost from new node to vertex is smaller
    #         G.cost[id_new] = distance_new_vert  # than current cost, rewire the vertex to new
    #         best_edge = (id_new, id_ver, distance_new_vert)

    return best_edge


def choose_parent(G: Graph, q_new: tuple, id_new: int, best_edge: tuple,
                  radius: float, obstacles: list, separate_tree_nodes: list = ()) -> tuple:
    """
    Find a node that is optimal in terms of cost to the start node.
    :param G: Graph
    :param q_new: position of new node
    :param id_new: id of new node
    :param best_edge: best edge so far
    :param radius: radius of search area
    :param obstacles: list of obstacles
    :param separate_tree_nodes: list if ids of nodes that are in the separate tree
    :return: id of best node
    """
    # costs = get_distance_dict(G, id_new)

    # for id_and_vertex, id_and_cost in zip(G.vertices.items(), costs.items()):
    #     # print(f"{id_and_vertex[0]}: {id_and_vertex[1]}, {id_and_cost[1]}")
    #     if id_and_vertex[0] in separate_tree_nodes: continue
    #     if id_and_vertex[0] == id_new: continue
    #     distance_new_vert = id_and_cost[1]
    #     if distance_new_vert > radius: continue
    #     line = Line(id_and_vertex[1], q_new)
    #     if through_obstacle(line, obstacles): continue
    #     if G.get_cost(id_new) < G.get_cost(id_and_vertex[0]) + distance_new_vert:
    #         G.cost[id_new] = distance_new_vert
    #         best_edge = (id_new, id_and_vertex[0], distance_new_vert)

    for id_ver, vertex in G.vertices.items():  # iterate through all the vertices
        if id_ver == id_new: continue
        distance_new_vert = calc_distance(q_new, vertex)  # calculate distance between new node and vertex node
        if distance_new_vert > radius: continue  # if distance is greater than search radius - continue
        line = Line(vertex, q_new)  # create Line object from new node to vertex
        if through_obstacle(line, obstacles): continue  # if the line goes through obstacle - continue
        if G.get_cost(id_new) > G.get_cost(id_ver) + distance_new_vert:  # if cost from new node to vertex is smaller
            G.cost[id_new] = distance_new_vert  # than current cost, rewire the vertex to new
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

    # costs = get_distance_dict(G, id_new)
    #
    # for id_and_vertex, id_and_cost in zip(G.vertices.items(), costs.items()):
    #     id_ver = id_and_vertex[0]
    #     vertex = id_and_vertex[1]
    #     distance_new_vert = id_and_cost[1]
    #
    #     if distance_new_vert > radius: continue
    #     if id_ver == G.id_vertex[G.start]: continue
    #     if id_ver == id_new: continue
    #     line = Line(vertex, q_new)
    #     if through_obstacle(line, obstacles): continue
    #     if G.get_cost(id_ver) > G.get_cost(id_new) + distance_new_vert:
    #         parent = G.parent[id_ver]  # parent of the rewired node
    #         del G.children[parent][G.children[parent].index(id_ver)]  # delete rewired node from it's parent children
    #         G.parent[id_ver] = id_new  # set rewired node's parent to new node
    #         G.children[id_new].append(id_ver)  # append rewired node to new node's children
    #         G.cost[id_ver] = distance_new_vert

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


def get_distance_dict(G: Graph, node_to_check: int, indeces_to_check: list[int]) -> dict:
    pos = G.vertices[node_to_check]
    tree_points_list = [vertex for id_ver, vertex in G.vertices.items()]

    tree_points = np.array(tree_points_list)
    new_point = np.array(pos).reshape(-1, 2)
    x2 = np.sum(tree_points ** 2, axis=1).reshape(-1, 1)
    y2 = np.sum(new_point ** 2, axis=1).reshape(-1, 1)
    xy = 2 * np.matmul(tree_points, new_point.T)
    dists = np.sqrt(x2 - xy + y2.T)

    distances = {id_ver: id_and_cost[0] for id_ver, id_and_cost in zip(G.vertices, dists)}

    return distances


def calc_distance(p1: tuple, p2: tuple) -> float:
    """
    Calculate distance between two points.
    :param p1: point 1
    :param p2: point 2
    :return: distance between points
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))
