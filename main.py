# from collections import deque
# from collections import defaultdict
# import heapq as heap
import warnings

import matplotlib.pyplot as plt
import random

from algorithm import RRT, RRT_star, RRT_star_FN
from graph import Graph, OutOfBoundsException
from map import Map

from algorithm import test_select_branch

from sdf_reader import SDFReader

NODE_RADIUS = 7

warnings.filterwarnings("error")

# because of matpltolib's bug
import matplotlib
matplotlib.use('TkAgg')

def plot_graph(graph: Graph, obstacles: list, xy_range: tuple = None):
    """
    Plot graph and obstacles.
    :param graph: Graph to plot
    :param obstacles: list of obstacles
    :param xy_range: tuple of x and y axes ranges
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

    if xy_range is not None:
        plt.xlim(xy_range[0][0], xy_range[0][1])
        plt.ylim(xy_range[1][0], xy_range[1][1])
    else:
        plt.xlim(0, graph.width)
        plt.ylim(0, graph.height)


if __name__ == '__main__':
    filepath = "D:/pwr-code/gazebo_model/turtlebot3_world_model.sdf"
    reader = SDFReader()
    reader.parse(filepath)
    obstacles, xy_range = reader.get_obstacles()

    map_width = 200
    map_height = 200
    start_node = (random.randint(xy_range[0][0], xy_range[0][1]), random.randint(xy_range[1][0], xy_range[1][1]))
    goal_node = (random.randint(xy_range[0][0], xy_range[0][1]), random.randint(xy_range[1][0], xy_range[1][1]))
    # start_node = (180, 130)
    # goal_node = (60, 140)
    my_map = Map((map_width, map_height), start_node, goal_node, NODE_RADIUS)
    # my_map.generate_obstacles(obstacle_count=40, size=7)
    # my_map.add_obstacles([[(60, 50), NODE_RADIUS], [(60, 60), NODE_RADIUS],
    #                       [(60, 70), NODE_RADIUS], [(60, 80), NODE_RADIUS],
    #                       [(60, 90), NODE_RADIUS], [(60, 100), NODE_RADIUS],
    #                       [(60, 110), NODE_RADIUS], [(60, 120), NODE_RADIUS],
    #                       [(70, 120), NODE_RADIUS], [(80, 120), NODE_RADIUS],
    #                       [(90, 120), NODE_RADIUS], [(100, 120), NODE_RADIUS],
    #                       [(110, 120), NODE_RADIUS], [(120, 120), NODE_RADIUS],
    #                       [(120, 50), NODE_RADIUS], [(120, 60), NODE_RADIUS],
    #                       [(120, 70), NODE_RADIUS], [(120, 80), NODE_RADIUS],
    #                       [(120, 90), NODE_RADIUS], [(120, 100), NODE_RADIUS],
    #                       [(120, 110), NODE_RADIUS], [(120, 120), NODE_RADIUS],
    #                       ]
    #                      )

    my_map.add_obstacles(obstacles * 100)

    # iteration = None
    # G = None

    # G = Graph(start_node, goal_node, map_width, map_height, xy_range)
    # iteration = RRT(G, iter_num=500, map=my_map, step_length=25, node_radius=NODE_RADIUS, bias=0)
    # plot_graph(G, my_map.obstacles_c, xy_range)
    # print(f"RRT algorithm stopped at iteration number: {iteration}")
    # plt.show()

    # G = Graph(start_node, goal_node, map_width, map_height, xy_range)
    # iteration, _ = RRT_star(G, iter_num=500, map=my_map, step_length=25, radius=70, node_radius=NODE_RADIUS, bias=0.01)
    # print(f"RRT_star algorithm stopped at iteration number: {iteration}")
    # plot_graph(G, my_map.obstacles_c, xy_range)
    # plt.show()

    try:
        G = Graph(start_node, goal_node, map_width, map_height, xy_range)
        iteration, _ = RRT_star_FN(G, iter_num=2000, map=my_map, step_length=35,
                                   radius=50, node_radius=NODE_RADIUS, max_nodes=50, bias=0.0, live_update=False)
    except OutOfBoundsException as e:
        pass
    print(f"RRT_star_FN algorithm stopped at iteration number: {iteration}")
    plot_graph(G, my_map.obstacles_c, xy_range)
    plt.show()

    # test_select_branch()

    plt.ioff()
    plt.show()
