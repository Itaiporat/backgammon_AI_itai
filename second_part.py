import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import copy
import time
import matplotlib.pyplot as plt

import Backgammon  # your game engine
import randomAgent  # used only as a placeholder if needed
from DeepRL_new.evaluation import evaluate_win_margin


# ----- Use pubeval_eval instead of assess_board -----
def flip_board(board):
    """
    Flips the board so that the perspective becomes that of player 1.
    This function uses a fixed index mapping (as in kotra's flipped_agent) to reverse the board.
    """
    idx = np.array([0, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13,
                    12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 26, 25, 28, 27])
    return -np.copy(board[idx])

def pubeval_eval(board):
    """
    Evaluates a board using Tesauro's pubeval.
    Flips the board as needed, converts it to int, and returns the evaluation.
    """
    from pubeval import pubeval_flip, lib, intp, israce
    board_flipped = pubeval_flip(board.copy())
    board_flipped = board_flipped.astype(np.int32)
    race = israce(board)
    value = lib.pubeval(race, board_flipped.ctypes.data_as(intp))
    return value

# ----- Update board_to_state to return 28 dimensions -----
def board_to_state(board):
    """
    Converts a Backgammon board (numpy array of length 29) into a 28-dimensional vector.
    - Takes board positions 1 to 24, normalizes them by dividing by 15 (24 values).
    - Concatenates positions 25 to 28 as extra features (4 values).
    Total dimension = 28.
    """
    positions = board[1:25].astype(np.float32) / 15.0  # 24 values
    extras = board[25:29].astype(np.float32)           # 4 values
    return np.concatenate((positions, extras))         # 28-dim vector

# ----- choose_best_move remains the same -----
def choose_best_move(board, dice, player, eval_function, num_candidates=4):
    """
    Returns the best move using the given evaluation function.
    It evaluates all legal moves and selects one of the top candidates using weighted probabilities.
    """
    moves, boards_after = Backgammon.legal_moves(board, dice, player)
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

# ----- train_model remains unchanged -----
def train_model(model, dataset, targets, num_epochs=20, batch_size=32, learning_rate=0.001):
    """
    Trains the network on the provided dataset using MSE loss.
    Returns a list of average training losses per epoch.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    dataset = torch.tensor(dataset, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    n_samples = dataset.shape[0]

    for epoch in range(num_epochs):
        permutation = torch.randperm(n_samples)
        epoch_loss = 0.0
        for i in range(0, n_samples, batch_size):
            indices = permutation[i:i + batch_size]
            batch_states = dataset[indices]
            batch_targets = targets[indices]
            optimizer.zero_grad()
            outputs = model(batch_states)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / (n_samples / batch_size)
        losses.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    return losses

def evaluate_relative_win(agent1, agent2, num_games=10):
    """
    Plays a series of games between agent1 and agent2.
    Returns the win rate of agent1 (as a percentage).
    Assumes that each agent object has an 'action' method.
    """
    wins = 0
    for i in range(num_games):
        winner, board = Backgammon.play_a_game(agent1, agent2)
        wins += evaluate_win_margin(board, winner)
        print(f"Evaluation Game {i + 1}/{num_games}: Winner = {'Agent1' if winner == 1 else 'Agent2'}")
    win_rate = (wins / num_games) * 100
    print(f"Agent1 Win Rate: {win_rate:.2f}% over {num_games} games.")
    return win_rate

def print_training_data_example(train_states, train_labels, model, num_examples=5):
    """
    Prints a few examples of the training data.
    For each example, shows the first 28 elements of the state vector,
    the training label, and the network's prediction.
    """
    print("\nExample Training Data:")
    for i in range(min(num_examples, len(train_states))):
        state = train_states[i]
        label = train_labels[i]
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        net_val = model(state_tensor).item()
        print(f"Example {i+1}:")
        print("  State vector (first 28 elements):", state[:28])
        print("  Training label:", label)
        print("  Network prediction:", net_val)


# ----- Agent Definitions -----
def new_network_agent_action(board, dice, player, i, model):
    """
    Uses the current network (model) to evaluate moves.
    If the board is not from player 1's perspective (i.e. if player != 1),
    the board is flipped before evaluation.
    """
    def net_eval(b):
        if player != 1:
            b = flip_board(b)
        state = board_to_state(b)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return model(state_tensor).item()
    return choose_best_move(board, dice, player, eval_function=net_eval)

def old_network_agent_action(board, dice, player, i, old_model):
    """
    Uses the old network snapshot (old_model) to evaluate moves.
    If the board is not from player 1's perspective, it is flipped before evaluation.
    """
    def old_net_eval(b):
        if player != 1:
            b = flip_board(b)
        state = board_to_state(b)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return old_model(state_tensor).item()
    return choose_best_move(board, dice, player, eval_function=old_net_eval)

class NewNetworkAgent:
    def __init__(self, model):
        self.model = model
    def action(self, board, dice, player, i):
        return new_network_agent_action(board, dice, player, i, self.model)

class OldNetworkAgent:
    def __init__(self, old_model):
        self.old_model = old_model
    def action(self, board, dice, player, i):
        return old_network_agent_action(board, dice, player, i, self.old_model)
class heuristic_agent:
    def __init__(self, heur = pubeval_eval):
        self.heur = heur
    def action(self, board_copy, dice, player, i):
        """
        Selects the best move using Tesauro's pubeval evaluation.
        If the board is not from player 1's perspective, it is flipped before evaluation.
        """
        moves, boards_after = Backgammon.legal_moves(board_copy, dice, player)
        if len(moves) == 0:
            return []  # No legal moves available.
        evals = []
        for b in boards_after:
            if player != 1:
                b = flip_board(b)
            evals.append(self.heur(b))
        best_index = np.argmax(evals)
        return moves[best_index]

# ----- Global Model -----
class HeuristicNet(nn.Module):
    def __init__(self, input_size=28, hidden_size=40):
        super(HeuristicNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

# Initialize global model and its snapshot.

model = HeuristicNet(input_size=28, hidden_size=40)
old_model = copy.deepcopy(model)

# ----- Self-Improvement Loop (for training on pubeval heuristic) -----
def self_improvement_loop_extended(run_duration=600, n_games_per_batch=20, batch_training_epochs=20,
                                   learning_rate=0.001, model=None, old_model=None):
    """
    Runs the self-improvement loop for a specified duration (in seconds).
    For each batch:
      - Runs n_games_per_batch between NewNetworkAgent (player 1) and heuristic_agent (using pubeval) (player -1).
      - Records every board (and mover) during the games.
      - Computes the win margin for each game using evaluate_win_margin.
      - Labels each board: label = margin if mover == winner, else 1 - margin.
      - Aggregates these examples, then trains the global model for batch_training_epochs.
      - Prints a few examples of the training data.
      - Evaluates the new network agent versus a fresh instance of heuristic_agent and records the win rate.
      - Also computes the average win margin of the games in this batch.
      - Updates the old network snapshot with the current model.
    Finally, plots the win rate and average win margin progression over batches,
    and saves the progress plot.
    """

    # Here, we use heuristic_agent for evaluation (which uses pubeval).



    new_agent = NewNetworkAgent(model)
    old_agent = OldNetworkAgent(old_model)

    training_buffer_states = []
    training_buffer_labels = []
    win_rates = []
    avg_win_margins = []
    batch_count = 0
    start_time = time.time()

    while time.time() - start_time < run_duration:
        batch_count += 1
        print(f"\n=== Batch {batch_count} ===")
        batch_win_margins = []
        for game in range(n_games_per_batch):
            game_boards = []  # list of tuples: (board, mover)
            board = Backgammon.init_board()
            current_player = 1
            previous_board = None
            while not Backgammon.game_over(board):
                game_boards.append((np.copy(board), current_player))
                dice = Backgammon.roll_dice()
                if current_player == 1:
                    move = new_agent.action(board, dice, current_player, 0)
                else:
                    move = old_agent.action(board, dice, current_player, 0)
                if len(move) == 0:
                    if previous_board is not None and np.array_equal(board, previous_board):
                        break
                    previous_board = np.copy(board)
                    # Switch turn and continue (pass move)
                    current_player = -current_player
                    continue
                if hasattr(move, "ndim") and move.ndim == 2:
                    for m in move:
                        board = Backgammon.update_board(board, m, current_player)
                else:
                    board = Backgammon.update_board(board, move, current_player)
                current_player = -current_player
            # Winner is -current_player.
            winner = -current_player
            margin = evaluate_win_margin(board, winner)
            batch_win_margins.append(margin)
            for (b, mover) in game_boards:
                label = margin
                training_buffer_states.append(board_to_state(b))
                training_buffer_labels.append(label)
        avg_margin = np.mean(batch_win_margins)
        avg_win_margins.append(avg_margin)
        print(f"Batch {batch_count}: Average Win Margin: {avg_margin:.4f}")

        train_states = np.array(training_buffer_states)
        train_labels = np.array(training_buffer_labels, dtype=np.float32).reshape(-1, 1)

        batch_losses = train_model(model, train_states, train_labels,
                                   num_epochs=batch_training_epochs,
                                   batch_size=32,
                                   learning_rate=learning_rate)
        print(f"Batch {batch_count} final loss: {batch_losses[-1]:.4f}")

        print_training_data_example(train_states, train_labels, model, num_examples=3)
        heur_agent = heuristic_agent()
        current_win_rate = evaluate_relative_win(new_agent, heur_agent, num_games=10)
        win_rates.append(current_win_rate)

        old_model = copy.deepcopy(model)
        old_agent = OldNetworkAgent(old_model)

        training_buffer_states = []
        training_buffer_labels = []

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, batch_count + 1), win_rates, marker='o')
    plt.xlabel("Batch Number")
    plt.ylabel("Win Rate (%)")
    plt.title("Win Rate Progression")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(range(1, batch_count + 1), avg_win_margins, marker='o', color='orange')
    plt.xlabel("Batch Number")
    plt.ylabel("Avg Win Margin")
    plt.title("Win Margin Progression")
    plt.grid(True)
    plt.tight_layout()
    # Save the plot to a file
    plt.savefig("progress_plots.png")
    print("Progress plots saved to 'progress_plots.png'.")
    plt.show()

    torch.save(model.state_dict(), "trained_model.pth")
    print("Model saved as 'trained_model.pth'.")


# ----- Run the Extended Self-Improvement Loop -----
if __name__ == '__main__':
    model = HeuristicNet(input_size=28, hidden_size=40)
    old_model = copy.deepcopy(model)
    # For testing, run for 300 seconds; adjust run_duration for longer runs.
    self_improvement_loop_extended(run_duration=600,
                                   n_games_per_batch=100,
                                   batch_training_epochs=32,
                                   learning_rate=0.001, model =model, old_model=old_model)
