import numpy as np
import math
from Backgammon import legal_moves  # Assuming Backgammon is in your PYTHONPATH


WHITE_EATEN_LOCATION = 26
BLACK_EATEN_LOCATION = 25
BLACK_OUT_LOCATION = 28
WHITE_OUT_LOCATION = 27
WHITE_HOME = 19
BLACK_HOME = 6
WHITE_START_IDX = 1
BLACK_START_IDX = 24

def assess_board(board):
    """
    Assess the board for the given player color based on the game's state.
    """
    weights_closed_game = {
        "location_rates": (location_rates, 0.4),
        "out_bonus": (out_bonus, 0.6),
    }

    weights_almost_closed = {
        "location_rates": (location_rates, 0.05),
        "houses_bonus": (houses_bonus, 0.3),
        "out_bonus": (out_bonus, 0.2),
        "mr_lonely_fine": (mr_lonely_fine, 0.35),
        "eaten_fine": (eaten_fine, 0.1),
    }

    weights_midgame = {
        "location_rates": (location_rates, 0.3),
        "eaten_fine": (eaten_fine, 0.25),
        "houses_bonus": (houses_bonus, 0.15),
        "mr_lonely_fine": (mr_lonely_fine, 0.1),
        "get_out_bonus": (get_out_bonus, 0.2)
    }


    # Define min and max values for normalization
    min_max = {
        "location_rates": (-22*15, 22*15),
        "eaten_fine": (-6, 6),
        "houses_bonus": (-128, 128),
        "out_bonus": (-15, 15),
        "mr_lonely_fine": (-15, 15),
        "get_out_bonus": (-2.5,2.5),
    }
    if is_game_closed(board):
        my_calculation = final_result(board, weights_closed_game, min_max)
    elif can_bear_off(board):
        #print("pieces: ", self.get_pieces(), "color: ")
        my_calculation = final_result(board, weights_almost_closed, min_max)
    else:
        my_calculation = final_result(board, weights_midgame, min_max)

    return my_calculation

def can_bear_off(board):
    return  ( all(x > 0 for x in board[WHITE_START_IDX: WHITE_HOME]) and board[WHITE_EATEN_LOCATION] == 0) or ( all(x > 0 for x in board[BLACK_HOME + 1:BLACK_START_IDX+1])  and board[BLACK_EATEN_LOCATION] == 0)


def location_rates(board):
    """Calculate the weighted location score of the pieces."""
    global WHITE_HOME, BLACK_START_IDX
    sum_locations_white = 0; sum_locations_black = 0
    for position in range(WHITE_START_IDX, BLACK_START_IDX + 1):
        if board[position] > 0:
            if WHITE_START_IDX <= position < WHITE_HOME:
                sum_locations_white += (board[position]) * position
            elif WHITE_HOME <=  position <= BLACK_START_IDX:
                sum_locations_white += (WHITE_HOME + 0.5 * (position - (WHITE_HOME))) * board[position]

        elif board[position] < 0:
            if BLACK_HOME <= position <= BLACK_START_IDX:
                sum_locations_black += (BLACK_START_IDX - position) * board[position]
            elif WHITE_START_IDX <= position <= BLACK_START_IDX:
                sum_locations_black += (19 + 0.5 * (5 - position)) * board[position]
    return sum_locations_white + sum_locations_black


def eaten_fine(board):
    """Calculate a penalty for eaten pieces."""
    return -board[WHITE_EATEN_LOCATION] + board[BLACK_EATEN_LOCATION] # White pieces on the bar


def houses_bonus(board):
    """Calculate the bonus for having connected houses."""
    white_bonus = 0; black_bonus = 0
    white_in_row = 0; black_in_row = 0

    for location in range(WHITE_HOME, BLACK_START_IDX +1):
        if board[location] > 1:
            white_in_row += 1
        else:
            white_bonus += 2 ** white_in_row
            white_in_row = 0
    if white_in_row > 0:
        white_bonus += 2 ** white_in_row

    for location in range(WHITE_START_IDX, BLACK_HOME + 1):
        if abs(board[location]) > 1:
            black_in_row += 1
        else:
            black_bonus -= 2 ** black_in_row
            black_in_row = 0
    if black_in_row > 0:
        black_bonus -= 2 ** black_in_row

    return white_bonus + black_bonus


def out_bonus(board):
    """Calculate the bonus for bearing off pieces."""
    return board[WHITE_OUT_LOCATION] - board[BLACK_OUT_LOCATION] # White pieces borne off



def mr_lonely_fine(board):
    """Penalty for leaving single pieces vulnerable."""
    white_fine = 0; black_fine = 0
    for location in range(WHITE_START_IDX, BLACK_START_IDX + 1):
        if board[location] == 1:
            white_fine -= 1
        elif board[location] == -1:
            black_fine += 1
    return white_fine + black_fine

def get_out_bonus(board):
    board = np.array(board)
    white_bonus =0
    black_bonus = 0
    sub_white = np.where(board[WHITE_START_IDX: BLACK_START_IDX + 1] > 0)[0]
    if len(sub_white) > 0:
        white_bonus = 0.5 * (np.min(sub_white))
    sub_black = np.where(board[WHITE_START_IDX: BLACK_START_IDX + 1] < 0)[0]
    if len(sub_black) > 0:
        black_bonus = -0.5 * (BLACK_START_IDX + 1 - np.max(sub_black))

    return white_bonus + black_bonus

def is_game_closed(board):
    sub_lst = board[WHITE_START_IDX: BLACK_START_IDX+1]  # Extract subarray from index 1 to 24
    found_positive = False  # Flag to track if we've encountered a negative number

    for num in sub_lst:
        if num > 0:
            found_positive = True  # After this, all numbers must be negative
        elif num < 0 and found_positive:
            return False  # Found a positive number after a negative -> invalid order

    return True  # If we never find a positive after a negative, return True


def final_result(board, weights_and_functions, min_max):
    """Normalize and combine weighted results from different functions."""
    result = 0
    for func_name, (func, weight) in weights_and_functions.items():
        min_val, max_val = min_max[func_name]
        value = func(board)
        normalized = normalize_and_weight(value, min_val, max_val, weight)
        result += normalized
    return result


def normalize_and_weight(value, min_val, max_val, weight):
    """Normalize a value and apply its weight."""
    normalized = (value - min_val) / (max_val - min_val)  # Normalize to [0, 1]
    return normalized * weight

def evaluate_win_margin(board, winner):
    """
    Computes a win-margin score based on the number of remaining pieces for the losing opponent.
    Returns a score between 0.6 (close win) and 1.0 (dominant win).

    Assumptions:
      - For player 1, borne-off checkers are at board[27].
      - For player -1, borne-off checkers are at board[28].
    """
    if winner == 1:
        borne_off_loser = board[28]
    else:
        borne_off_loser = board[27]
    remaining = 15 - abs(borne_off_loser)
    margin_score = 0.6 + 0.4 * (remaining / 15.0)
    return margin_score if winner == 1 else 1 - margin_score


def board_to_state(board):
    """
    Converts a Backgammon board (numpy array of length 29) into a normalized state vector of dimension 28.
    Uses board[1:25] (the board positions) normalized by dividing by 15.0,
    and appends the extra features board[25:29] (e.g. dead pieces, borne-off counts).
    """
    positions = board[1:25].astype(np.float32) / 15.0  # 24 elements
    extras = board[25:29].astype(np.float32)  # 4 elements
    return np.concatenate((positions, extras))  # total = 28


def choose_best_move(board, dice, player, eval_function, num_candidates=4):
    """
    Computes all legal moves from the given board state, evaluates each resulting board
    using eval_function, and selects one of the top moves using weighted probabilities.
    """
    moves, boards_after = legal_moves(board, dice, player)
    if len(moves) == 0:
        return []  # No legal moves available.
    evaluations = [eval_function(b) for b in boards_after]
    sorted_indices = np.argsort(evaluations)[::-1]
    top_indices = sorted_indices[:num_candidates] if len(moves) >= num_candidates else sorted_indices
    top_moves = [moves[i] for i in top_indices]
    top_values = [evaluations[i] for i in top_indices]
    weights = np.array([math.exp(v) for v in top_values])
    norm_weights = weights / np.sum(weights)
    chosen_index = np.random.choice(len(top_moves), p=norm_weights)
    return top_moves[chosen_index]
