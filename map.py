import numpy as np
import random
import matplotlib.pyplot as plt


class Map:
    def __init__(self, size: tuple, start: tuple, goal: tuple, node_radius: int):
        """
        :param size: width and height of the map
        :param start: position of start node
        :param goal: position of goal node
        """
        self.obstacles_c = []  # list of obstacles' position and radius (circles)
        self.obstacles_r = []  # list of obstacles' upper-left and lower-right position (rectangles)
        self.width = size[0]
        self.height = size[1]

        self.start = start
        self.goal = goal

        self.node_radius = node_radius

    def collides_w_start_goal_c(self, obs) -> bool:
        """ Check if circular obstacle collides with start or goal node
        :param obs: list of obstacle positions and radia
        :return: True if collision detected, False if collision not detected
        """
        distance_start = np.sqrt((obs[0][0] - self.start[0]) ** 2 + (obs[0][1] - self.start[1]) ** 2)
        distance_goal = np.sqrt((obs[0][0] - self.goal[0]) ** 2 + (obs[0][1] - self.goal[1]) ** 2)
        if distance_start < self.node_radius + obs[1] or distance_goal < self.node_radius + obs[1]:
            return True
        return False

    def is_occupied_c(self, p: tuple) -> bool:
        """ Check if space is occupied by circular obstacle
        :param p: position of point to check
        :return: True if occupied, False if not occupied
        """
        for obs in self.obstacles_c:
            distance = np.sqrt((p[0] - obs[0][0])**2 + (p[1] - obs[0][1])**2)
            if distance < self.node_radius + obs[1]:
                return True
        return False

    def generate_obstacles(self, obstacle_count: int = 10, size: int = 1):
        """ Generate random obstacles stored in self.obstacles. Rectangles not yet implemented
        :param obstacle_count: number of obstacles to generate
        :param size: radius of obstacles
        """
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
