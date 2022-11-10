import random


class OutOfBoundsException(Exception):
    print("Start or goal point out of bounds.")


class Graph:
    def __init__(self, start: tuple, goal: tuple, width: int, height: int):
        """
        :param start:
        :param goal:
        :param width:
        :param height:
        """
        if (start[0] or goal[0]) > width or (start[0] or goal[0]) < 0 or \
                (start[1] or goal[1]) > height or (start[1] or goal[1]) < 0:
            raise OutOfBoundsException()

        self.start = start  # start position
        self.goal = goal  # goal position
        self.width = width
        self.height = height

        self.parent = {0: None}
        self.children = {0: []}
        self.vertices = {0: self.start}
        self.cost = {0: 0.0}        # to node's parent
        self.id_vertex = {self.start: 0}

        self.last_id = 0

    def get_cost(self, from_node: int) -> float:
        """
        Get cost to start_node
        :return:
        :param from_node: id of node from which to calculate the cost
        :return: cost
        """
        cost_list = []
        node = from_node
        while self.parent[node] is not None:
            cost_list.append(self.cost[node])
            node = self.parent[node]
        return sum(cost_list)

    def add_vertex(self, pos: tuple) -> int:
        """
        Add new vertex to the Graph
        :param pos: position of the new vertex
        :return: id of the new vertex
        """
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

    def remove_vertex(self, id: int):
        """
        Remove vertex from the tree
        :param id: id of node to remove
        """
        parent = self.parent[id]
        pos = self.vertices[id]
        if self.parent[id] is not None:
            del self.children[parent][self.children[parent].index(id)]
        del self.parent[id]
        del self.children[id]
        del self.vertices[id]
        del self.cost[id]
        del self.id_vertex[pos]
        # del self.edges[id]
        # del self.neighbors[id]

    def add_edge(self, id_node: int, id_parent: int, cost: float):
        """ Add new edge to the graph
        :param id_node: id of child node
        :param id_parent: id of parent node
        :param cost: cost of the edge between two nodes
        """
        self.parent[id_node] = id_parent
        self.children[id_parent].append(id_node)
        self.cost[id_node] = cost

    def random_node(self, bias: float = 0) -> tuple:
        """
        Generate random point on the map, if bias is between [0;1),
        then there's a probability that point will be the goal point
        :param bias: 0-1, possibility that direction of new node will be goal node
        :return: position of node
        """
        if random.random() < bias:
            return self.goal
        return random.randint(2, self.width), random.randint(2, self.height)

