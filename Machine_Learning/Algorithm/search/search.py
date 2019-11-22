def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    # raise NotImplementedError
    if start == goal:
        return []
    count = 1
    current_loc = start
    explored = set()
    path = [start]
    frontiers = PriorityQueue()
    depth = 0
    frontiers.append((depth, count, current_loc, path))
    found = False
    # loop = 0
    while not found:
        # print loop
        # loop += 1
        if frontiers.size() == 0:
            return 'fail to find the path'
        node = frontiers.pop()
        explored.add(node[2])
        neighbors = graph.neighbors(node[2])
        depth += 1
        for current_loc in neighbors:
            if current_loc not in explored and current_loc not in frontiers:
                new_path = deepcopy(node[3])
                new_path.append(current_loc)
                if current_loc == goal:
                    return new_path
                count += 1
                frontiers.append((depth, count, current_loc, new_path))


def uniform_cost_search(graph, start, goal):
    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    # raise NotImplementedError
    if start == goal:
        return []
    count = 1
    current_loc = start
    explored = set()
    path = [start]
    frontiers = PriorityQueue()
    distance = 0.0
    frontiers.append((distance, count, current_loc, path))
    found = False
    # loop = 0
    while not found:
        # print loop
        # loop += 1
        if frontiers.size() == 0:
            return 'fail to find the path'
        node = frontiers.pop()
        if node[2] == goal:
            return node[3]
        explored.add(node[2])
        neighbors = graph.neighbors(node[2])
        for current_loc in neighbors:
            # print 'node', node
            if current_loc not in explored and current_loc not in frontiers:
                distance = node[0] + graph[node[2]][current_loc]['weight']
                new_path = deepcopy(node[3])
                new_path.append(current_loc)
                # print current_loc, goal, new_path
                count += 1
                frontiers.append((distance, count, current_loc, new_path))
            elif current_loc in frontiers:
                existing_node = frontiers.get(current_loc)
                distance = node[0] + graph[node[2]][current_loc]['weight']
                if existing_node[0] > distance:
                    frontiers.remove(current_loc)
                    new_path = deepcopy(node[3])
                    new_path.append(current_loc)
                    count += 1
                    frontiers.append((distance, count, current_loc, new_path))


def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """

    # TODO: finish this function!
    # raise NotImplementedError
    pos_v = graph.node[v]['pos']
    pos_g = graph.node[goal]['pos']
    return math.sqrt((pos_v[0] - pos_g[0])**2 + (pos_v[1] - pos_g[1])**2)


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    # raise NotImplementedError
    if start == goal:
        return []
    count = 1
    current_loc = start
    explored = set()
    path = [start]
    frontiers = PriorityQueue()
    f = heuristic(graph, start, goal)
    frontiers.append((f, count, current_loc, path))
    found = False
    # loop = 0
    while not found:
        # print loop
        # loop += 1
        if frontiers.size() == 0:
            return 'fail to find the path'
        node = frontiers.pop()
        explored.add(node[2])
        # print 'explored', explored
        neighbors = graph.neighbors(node[2])
        for current_loc in neighbors:
            if node[2] == goal:
                return node[3]
            # print 'node', node
            if current_loc not in explored and current_loc not in frontiers:
                # print node
                # print node[3]
                f = node[0] - heuristic(graph, node[2], goal) + \
                    graph[node[2]][current_loc]['weight'] + \
                    heuristic(graph, current_loc, goal)
                new_path = deepcopy(node[3])
                new_path.append(current_loc)
                # print current_loc, goal, new_path
                count += 1
                # print graph[node[2]][current_loc]['weight']
                # print (f, count, current_loc, new_path)
                frontiers.append((f, count, current_loc, new_path))
            elif current_loc in frontiers:
                existing_node = frontiers.get(current_loc)
                f = node[0] - heuristic(graph, node[2], goal) + \
                    graph[node[2]][current_loc]['weight'] + \
                    heuristic(graph, current_loc, goal)
                # print 'existing node', existing_node
                # print distance
                if existing_node[0] > f:
                    frontiers.remove(current_loc)
                    new_path = deepcopy(node[3])
                    new_path.append(current_loc)
                    count += 1
                    # print 'replaced by', (f, count, current_loc, new_path)
                    frontiers.append((f, count, current_loc, new_path))

def check_joint(forward, backward, shortest_path, Astar):
    # print 'explored_f', explored_f
    # print 'explored_b', explored_b
    for _, _, v, _ in forward:
        if v in backward:
            node_f = forward.get(v)
            node_b = backward.get(v)
            if Astar:
                distance = node_f[1] + node_b[1]
            else:
                distance = node_f[0] + node_b[0]
            if distance < shortest_path['mu']:
                shortest_path['mu'] = distance
                shortest_path['path'] = node_f[3][:-1] + node_b[3][::-1]
                # print shortest_path
    return shortest_path

def check_termination(frontiers_f, frontiers_b, explored_f, explored_b,shortest_path, Astar=False, pr_t=0.0):
    shortest_path = check_joint(frontiers_f, explored_b, shortest_path, Astar)
    if frontiers_f.top()[0] + frontiers_b.top()[0] >= shortest_path['mu'] + pr_t:
        return True, shortest_path
    shortest_path = check_joint(explored_f, frontiers_b, shortest_path, Astar)
    if frontiers_f.top()[0] + frontiers_b.top()[0] >= shortest_path['mu'] + pr_t:
        return True, shortest_path
    return False, shortest_path



def ucs_helper(graph, frontiers, explored, count):
    node = frontiers.pop()
    explored.append(node)
    neighbors = graph.neighbors(node[2])
    for current_loc in neighbors:
        # print 'node', node
        if current_loc not in explored and current_loc not in frontiers:
            distance = node[0] + graph[node[2]][current_loc]['weight']
            new_path = deepcopy(node[3])
            new_path.append(current_loc)
            # print current_loc, goal, new_path
            count += 1
            frontiers.append((distance, count, current_loc, new_path))
        elif current_loc in frontiers:
            existing_node = frontiers.get(current_loc)
            distance = node[0] + graph[node[2]][current_loc]['weight']
            if existing_node[0] > distance:
                frontiers.remove(current_loc)
                new_path = deepcopy(node[3])
                new_path.append(current_loc)
                count += 1
                # print 'replaced by', (distance, count, current_loc, new_path)
                frontiers.append((distance, count, current_loc, new_path))
    return frontiers, explored, count

def bidirectional_ucs(graph, start, goal):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    # TODO: finish this function!
    # raise NotImplementedError

    if start == goal:
        return []
    count_f = 1
    count_b = 1
    explored_f = PriorityQueue()
    explored_b = PriorityQueue()
    frontiers_f = PriorityQueue()
    frontiers_b = PriorityQueue()
    frontiers_f.append((0.0, count_f, start, [start]))
    frontiers_b.append((0.0, count_b, goal, [goal]))
    shortest_path = {'mu': float("inf"), 'path':[]}
    found = False
    # loop = 0
    while not found:
        # loop += 1
        # print loop
        if frontiers_f.size() == 0 or frontiers_b.size() == 0:
            return 'fail to find the path'
        frontiers_f, explored_f, count_f = ucs_helper(graph, frontiers_f, explored_f, count_f)
        is_terminal, shortest_path = check_termination(frontiers_f, frontiers_b, explored_f, explored_b,shortest_path)
        if is_terminal:
            return shortest_path['path']
        # print shortest_path
        frontiers_b, explored_b, count_b = ucs_helper(graph, frontiers_b, explored_b, count_b)
        is_terminal, shortest_path = check_termination(frontiers_f, frontiers_b, explored_f, explored_b,shortest_path)
        if is_terminal:
            return shortest_path['path']


def p_f(graph, v, end_points, heuristic = euclidean_dist_heuristic):
    return (heuristic(graph,v, end_points[1]) - heuristic(graph,end_points[0], v))/2.0 + \
            heuristic(graph, end_points[1], end_points[0])/2.0

def p_r(graph, v, end_points, heuristic = euclidean_dist_heuristic):
    return (heuristic(graph,end_points[0], v) - heuristic(graph,v, end_points[1]))/2.0 + \
            heuristic(graph, end_points[0], end_points[1])/2.0


def Astar_helper(graph, start, goal, frontiers, explored, heuristic = p_f):

    node = frontiers.pop()
    explored.append(node)
    # print 'explored', explored
    neighbors = graph.neighbors(node[2])
    for current_loc in neighbors:
        if current_loc not in explored and current_loc not in frontiers:
            distance = node[1] + graph[node[2]][current_loc]['weight']
            f = distance + heuristic(graph, current_loc, (start, goal))
            new_path = deepcopy(node[3])
            new_path.append(current_loc)
            # print f, distance, current_loc, new_path
            frontiers.append((f, distance, current_loc, new_path))
        elif current_loc in frontiers:
            existing_node = frontiers.get(current_loc)
            distance = node[1] + graph[node[2]][current_loc]['weight']
            f = distance + heuristic(graph, current_loc, (start, goal))
            if existing_node[0] > f:
                frontiers.remove(current_loc)
                new_path = deepcopy(node[3])
                new_path.append(current_loc)
                frontiers.append((f, distance, current_loc, new_path))
    return frontiers, explored
