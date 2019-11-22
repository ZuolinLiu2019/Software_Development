def tridirectional_search(graph, goals):
    """
    Exercise 3: Tridirectional UCS Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    # raise NotImplementedError
    if goals[0] == goals[1] == goals[2]:
        path = []
        return path
    num_points = len(goals)
    for i in range(num_points):
        if goals[i%num_points] == goals[(i+1)%num_points]:
            path = bi_ucs(graph, goals[(i+1)%num_points], goals[i-1])
            return path

    shortest_paths = [{'mu': float("inf"), 'path':[]},
                        {'mu': float("inf"), 'path':[]},
                        {'mu': float("inf"), 'path':[]}]

    found = [False, False, False]
    valid_path = [False, False, False]

    explored = []
    frontiers = []
    count = [1, 1, 1]

    for i in range(num_points):
        explored.append(PriorityQueue())
        frontiers.append(PriorityQueue())
        frontiers[i].append((0.0, count[i], goals[i], [goals[i]]))

    terminate = False

    while not terminate and not all(found):
        # determine which node to explore
        if sum(valid_path) == 2:
            for x in range(len(valid_path)):
                if not valid_path[x]:
                    distance = max(shortest_paths[(x+1)%num_points]['mu'], shortest_paths[x-1]['mu'])
                    if frontiers[x].top()[0] + frontiers[(x+1)%num_points].top()[0] < distance:
                        if frontiers[x].top()[0] < frontiers[(x+1)%num_points].top()[0]:
                            i = x
                        else:
                            i = (x+1)%num_points
                    else:
                        terminate = True

        else:
            for k in range(num_points):
                if not found[k] or not found[k-1]:
                    min_potential = frontiers[k].top()[0]
                    i = k
                    break
            for j in range(k + 1, num_points):
                if frontiers[j].top()[0] < min_potential and (not found[j] or not found[j-1]):
                    min_potential = frontiers[j].top()[0]
                    i = j
        if not found[i] or not found[i-1]:
            # no node to explore
            if frontiers[i].size() == 0:
                found[i] = True
                found[i-1] = True
            # explore the frontier node
            frontiers[i], explored[i], count[i] = ucs_helper(graph,frontiers[i], explored[i], count[i])
            if not found[i]:
                is_terminal, shortest_paths[i] = check_termination(frontiers[i], frontiers[(i+1)%num_points],
                                                    explored[i], explored[(i+1)%num_points], shortest_paths[i])

                if is_terminal:
                    found[i] = True
                    valid_path[i] = True
            if not found[i-1]:
                is_terminal, shortest_paths[i-1] = check_termination(frontiers[i-1], frontiers[i],
                                                     explored[i-1], explored[i],  shortest_paths[i-1])
                if is_terminal:
                    found[i-1] = True
                    valid_path[i-1] = True
    # combine a final path
    final_path = connect_route(shortest_paths, valid_path)
    return final_path


def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic):
    """
    Exercise 3: Upgraded Tridirectional Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    # raise NotImplementedError

    if goals[0] == goals[1] == goals[2]:
        path = []
        return path
    num_points = len(goals)
    for i in range(num_points):
        if goals[i%num_points] == goals[(i+1)%num_points]:
            path = bi_a_star(graph, goals[(i+1)%num_points], goals[i-1])
            return path
    shortest_paths = [{'mu': float("inf"), 'path':[]},
                        {'mu': float("inf"), 'path':[]},
                        {'mu': float("inf"), 'path':[]}]

    found = [False, False, False]
    valid_path = [False, False, False]

    explored = []
    frontiers = []
    pr_t = []
    # route = []

    for i in range(num_points):
        explored.append([])
        frontiers.append([])
        pr_t.append([])
        start = goals[i]
        # route.append([])
        for j in range (num_points - 1):
            goal = goals[(i+j+1)%num_points]
            explored[i].append(PriorityQueue())
            frontiers[i].append(PriorityQueue())
            pr_t[i].append(p_r(graph, goal, (start, goal)))
            f_f = p_f(graph, start, (start, goal))
            frontiers[i][j].append((f_f, 0.0, start, [start]))
            # route[i].append((start, goal))

    # print pr_t
    # print route
    terminate = False

    while not terminate and not all(found):
        # determine which node to explore
        if sum(valid_path) == 2:
            for x in range(len(valid_path)):
                if not valid_path[x]:
                    distance = max(shortest_paths[(x+1)%num_points]['mu'], shortest_paths[x-1]['mu'])
                    if frontiers[x][0].top()[0] + frontiers[(x+1)%num_points][1].top()[0] < distance + pr_t[x][0]:
                        if frontiers[x][0].top()[0] < frontiers[(x+1)%num_points][1].top()[0]:
                            i = x
                            j = 0
                        else:
                            i = (x+1)%num_points
                            j = 1

                    else:
                        terminate = True

        else:
            for y in range(num_points):
                if not found[y]:
                    i = y
                    j = 0
                    min_potential = frontiers[y][0].top()[0]
                    break
                if not found[y-1]:
                    i = y
                    j = 1
                    min_potential = frontiers[y][1].top()[0]
                    break
            # print i, j
            for z in range(num_points):
                if frontiers[z][0].top()[0] < min_potential and not found[z]:
                    min_potential = frontiers[z][0].top()[0]
                    i = z
                    j = 0
                if frontiers[z][1].top()[0] < min_potential and  not found[z-1]:
                    min_potential = frontiers[z][1].top()[0]
                    i = z
                    j = 1
        # print i, j
        if not found[i] or not found[i-1]:
            # no node to explore
            if frontiers[i][j].size() == 0 and j == 0:
                found[i] = True
            if frontiers[i][j].size() == 0 and j == 1:
                found[i-1] = True
            # explore the frontier node
            # print frontiers[i][j]
            frontiers[i][j], explored[i][j] = Astar_helper(graph, goals[i], goals[(i+j+1)%num_points], frontiers[i][j], explored[i][j], heuristic = p_f)
            # print 'route', route[i][j]
            # print 'forward',frontiers[i][j]
            # print 'backward', frontiers[(i+j+1)%num_points][1-j]
            if not found[i-j]:
                is_terminal, shortest_paths[i-j] = check_termination(frontiers[i][j], frontiers[(i+j+1)%num_points][1-j],
                                                    explored[i][j], explored[(i+j+1)%num_points][1-j], shortest_paths[i-j], Astar=True, pr_t = pr_t[i][j])
                # print shortest_paths
                if is_terminal:
                    found[i-j] = True
                    valid_path[i-j] = True
                # print valid_path
    # combine a final path
    # print 'shortest_paths', shortest_paths
    final_path = connect_route(shortest_paths, valid_path)
    # print'final path', final_path
    return final_path
