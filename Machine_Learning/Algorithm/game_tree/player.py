from isolation import Board, game_as_text
from random import randint

class OpenMoveEvalFn:

    def score(self, game, maximizing_player_turn=True):
        """Score the current game state

        Evaluation function that outputs a score equal to how many
        moves are open for AI player on the board minus how many moves
        are open for Opponent's player on the board.
        Note:
            1. Be very careful while doing opponent's moves. You might end up
               reducing your own moves.
            3. If you think of better evaluation function, do it in CustomEvalFn below.

            Args
                param1 (Board): The board and game state.
                param2 (bool): True if maximizing player is active.

            Returns:
                float: The current state's score. MyMoves-OppMoves.

            """

        num_moves1 = len(game.get_legal_moves())
        num_moves2 = len(game.get_opponent_moves())
        if maximizing_player_turn:
            if num_moves1 == 0:
                return float("-1000")
            elif num_moves2 == 0:
                return float("1000")
            return num_moves1 - num_moves2
        else:
            if num_moves1 == 0:
                return float("1000")
            elif num_moves2 == 0:
                return float("-1000")
            return num_moves2 - num_moves1


class CustomEvalFn:
    def __init__(self):
        pass

    def score(self, game, maximizing_player_turn=True):
        """Score the current game state

        Custom evaluation function that acts however you think it should. This
        is not required but highly encouraged if you want to build the best
        AI possible.

        Args
            game (Board): The board and game state.
            maximizing_player_turn (bool): True if maximizing player is active.

        Returns:
            float: The current state's score, based on your own heuristic.

        """

        # # TODO: finish this function!
        raise NotImplementedError

class CustomPlayer:
    """Player that chooses a move using your evaluation function
    and a minimax algorithm with alpha-beta pruning.
    You must finish and test this player to make sure it properly
    uses minimax and alpha-beta to return a good move."""

    def __init__(self, search_depth=2, eval_fn=OpenMoveEvalFn()):
        """Initializes your player.

        if you find yourself with a superior eval function, update the default
        value of `eval_fn` to `CustomEvalFn()`

        Args:
            search_depth (int): The depth to which your agent will search
            eval_fn (function): Utility function used by your agent
        """
        self.eval_fn = eval_fn
        self.search_depth = search_depth

    def move(self, game, legal_moves, time_left):
        """Called to determine one move by your agent

            Note:
                1. Do NOT change the name of this 'move' function. We are going to call
                the this function directly.
                2. Change the name of minimax function to alphabeta function when
                required. Here we are talking about 'minimax' function call,
                NOT 'move' function name.
                Args:
                game (Board): The board and game state.
                legal_moves (list): List of legal moves
                time_left (function): Used to determine time left before timeout

            Returns:
                tuple: best_move
            """

        # best_move, utility = self.minimax(game, time_left, 0)
        best_move, utility = self.alphabeta(game, time_left, 0)
        # print best_move, utility
        return best_move

    def utility(self, game, maximizing_player):
        """Can be updated if desired. Not compulsory. """
        return self.eval_fn.score(game, maximizing_player_turn=maximizing_player)

    def IsTermial(self, game):
        if len(game.get_legal_moves()) == 0:
            return True
        else:
            return False

    def minimax(self, game, time_left, depth, maximizing_player=True):
        """Implementation of the minimax algorithm

        Args:
            game (Board): A board and game state.
            time_left (function): Used to determine time left before timeout
            depth: Used to track how deep you are in the search tree
            maximizing_player (bool): True if maximizing player is active.

        Returns:
            (tuple, int): best_move, val
        """
        def max_value(game, last_move, depth, maximizing_player_turn = True):
            if self.IsTermial(game) or depth >= self.search_depth:
                return self.utility(game, maximizing_player_turn), last_move
            else:
                max_v = float("-inf")
                best_move = None
                moves = game.get_legal_moves()
                depth += 1
                for move in moves:
                    next_game = game.copy()
                    is_over, winner = next_game.__apply_move__(move)
                    if is_over:
                        if depth == 1:
                            return float("1000"), move
                        v = float("1000")
                    else:
                        v, _= min_value(next_game, move, depth)
                    # print max_v, v
                    if max_v < v:
                        max_v = v
                        best_move = move
            return max_v, best_move

        def min_value(game, last_move, depth, maximizing_player_turn = False):
            if self.IsTermial(game) or depth >= self.search_depth:
                return self.utility(game, maximizing_player_turn), last_move
            else:
                min_v = float("inf")
                best_move = None
                moves = game.get_legal_moves()
                depth += 1
                for move in moves:
                    next_game = game.copy()
                    is_over, winner = next_game.__apply_move__(move)
                    if is_over:
                        v = float("-1000")
                    else:
                        v, _= max_value(next_game, move, depth)
                    # print min_v, v
                    if min_v > v:
                        min_v = v
                        best_move = move
            return min_v, best_move


        game_copy = game.copy()
        # tree = []
        best_val, best_move = max_value(game_copy, None, depth)
        return best_move, best_val


    def alphabeta(self, game, time_left, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implementation of the alphabeta algorithm

        Args:
            game (Board): A board and game state.
            time_left (function): Used to determine time left before timeout
            depth: Used to track how deep you are in the search tree
            alpha (float): Alpha value for pruning
            beta (float): Beta value for pruning
            maximizing_player (bool): True if maximizing player is active.

        Returns:
            (tuple, int): best_move, val
        """

        def search(game, alpha, beta, depth, time_left):
            def max_value(game, last_move, depth, alpha, beta, time_left, maximizing_player_turn = True):
                if time_left() < 10:
                    raise RuntimeError
                if self.IsTermial(game) or depth >= self.search_depth:
                    return self.utility(game, maximizing_player_turn), last_move
                max_v = float("-inf")
                best_move = None
                moves = game.get_legal_moves()
                depth += 1
                for move in moves:
                    next_game = game.copy()
                    is_over, winner = next_game.__apply_move__(move)
                    if is_over:
                        if depth == 1:
                            return float("inf"), move
                        v = float("1000")
                    else:
                        v, _ = min_value(next_game, move, depth, alpha, beta, time_left)
                    if max_v < v:
                        max_v = v
                        best_move = move
                    if max_v >= beta:
                        return max_v, best_move
                    alpha = max(alpha, max_v)
                return max_v, best_move
            def min_value(game, last_move, depth, alpha, beta, time_left, maximizing_player_turn = False):
                # print time_left()
                if time_left() < 10:
                    raise RuntimeError
                if self.IsTermial(game) or depth >= self.search_depth:
                    return self.utility(game, maximizing_player_turn), last_move
                # else:
                min_v = float("inf")
                best_move = None
                moves = game.get_legal_moves()
                # print moves
                depth += 1
                for move in moves:
                    next_game = game.copy()
                    is_over, winner = next_game.__apply_move__(move)
                    # print 'min', is_over
                    if is_over:
                        v = float("-1000")
                    else:
                        v, _, = max_value(next_game, move, depth, alpha, beta, time_left)
                    if min_v > v:
                        min_v = v
                        best_move = move
                    if min_v <= alpha:
                        return min_v, best_move
                    beta = min(beta, min_v)
                return min_v, best_move

            game_copy = game.copy()
            # tree = []
            best_val, best_move= max_value(game_copy, None, depth, alpha, beta, time_left)
            return best_move, best_val

        if self.IsTermial(game):
            return None, self.utility(game, maximizing_player=True)

        if (int(game.height/2), int(game.height/2), False) in game.get_legal_moves():
            return (int(game.height/2), int(game.height/2), False), 0

        self.search_depth = 1
        best_move = None
        best_val = float("-inf")
        for i in range(10):s
            game_copy = game.copy()
            try:
                m, v = search(game_copy, alpha, beta, depth, time_left)
            except RuntimeError:
                return best_move, best_val
            best_move = m
            best_val = v
            self.search_depth += 1
            # else:
            #     return best_move, best_val
        return best_move, best_val
