def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.

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
    explored_f = PriorityQueue()
    explored_b = PriorityQueue()
    frontiers_f = PriorityQueue()
    frontiers_b = PriorityQueue()
    pr_t = p_r(graph, goal, (start, goal))
    f_f = p_f(graph, start, (start, goal))
    f_b = p_r(graph, goal, (start, goal))
    frontiers_f.append((f_f, 0.0, start, [start]))
    frontiers_b.append((f_b, 0.0, goal, [goal]))
    shortest_path = {'mu': float("inf"), 'path':[]}
    found = False
    # loop = 0
    while not found:
        # loop += 1
        # print loop
        if frontiers_f.size() == 0 or frontiers_b.size() == 0:
            return 'fail to find the path'
        frontiers_f, explored_f = Astar_helper(graph, start, goal, frontiers_f, explored_f, heuristic = p_f)
        is_terminal, shortest_path = check_termination(frontiers_f, frontiers_b, explored_f, explored_b,shortest_path, Astar=True, pr_t = pr_t)
        if is_terminal:
            return shortest_path['path']
        frontiers_b, explored_b = Astar_helper(graph, start, goal, frontiers_b, explored_b, heuristic = p_r)
        is_terminal, shortest_path = check_termination(frontiers_f, frontiers_b, explored_f, explored_b,shortest_path, Astar=True, pr_t = pr_t)
        if is_terminal:
            return shortest_path['path']


def bi_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.

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
    explored_f = PriorityQueue()
    explored_b = PriorityQueue()
    frontiers_f = PriorityQueue()
    frontiers_b = PriorityQueue()
    pr_t = p_r(graph, goal, (start, goal))
    f_f = p_f(graph, start, (start, goal))
    f_b = p_r(graph, goal, (start, goal))
    frontiers_f.append((f_f, 0.0, start, [start]))
    frontiers_b.append((f_b, 0.0, goal, [goal]))
    shortest_path = {'mu': float("inf"), 'path':[]}
    found = False
    # loop = 0
    while not found:
        # loop += 1
        # print loop
        if frontiers_f.size() == 0 or frontiers_b.size() == 0:
            return 'fail to find the path'
        frontiers_f, explored_f = Astar_helper(graph, start, goal, frontiers_f, explored_f, heuristic = p_f)
        is_terminal, shortest_path = check_termination(frontiers_f, frontiers_b, explored_f, explored_b,shortest_path, Astar=True, pr_t = pr_t)
        if is_terminal:
            return shortest_path['path']
        frontiers_b, explored_b = Astar_helper(graph, start, goal, frontiers_b, explored_b, heuristic = p_r)
        is_terminal, shortest_path = check_termination(frontiers_f, frontiers_b, explored_f, explored_b,shortest_path, Astar=True, pr_t = pr_t)
        if is_terminal:
            return shortest_path['path']

def bi_ucs(graph, start, goal):
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

def connect_route(shortest_paths, valid_path):
    if sum(valid_path) == 3:
        paths = optimal_path(shortest_paths)
    else:
        paths = []
        for i in range(len(valid_path)):
            if len(shortest_paths[i]['path']) > 0 and valid_path[i]:
                paths.append(shortest_paths[i]['path'])
    # print 'paths', paths
    if len(paths) == 0:
        return []
    elif len(paths) == 1:
        return paths[0]
    elif len(paths) == 2:
        if paths[0][0] == paths[1][-1]:
            return paths[1][:-1] + paths[0]
        elif paths[0][-1] == paths[1][0]:
            return paths[0][:-1] + paths[1]
        elif paths[0][0] == paths[1][0]:
            return paths[0][::-1] + paths[1][1:]
        elif paths[0][-1] == paths[1][-1]:
            # print paths[1][:-1] + paths[0][::-1]
            return paths[1][:-1] + paths[0][::-1]

def optimal_path(shortest_paths):
    max_distance = shortest_paths[0]['mu']
    max_path = shortest_paths[0]['path']
    paths = []
    for i in range(1, len(shortest_paths)):
        if shortest_paths[i]['mu'] < max_distance:
            if len(shortest_paths[i]['path']) > 0:
                paths.append(shortest_paths[i]['path'])
        else:
            if len(max_path) > 0:
                paths.append(max_path)
            max_distance = shortest_paths[i]['mu']
            max_path = shortest_paths[i]['path']
    return paths
