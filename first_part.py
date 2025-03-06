import numpy as np
import torch
import matplotlib.pyplot as plt
from model import HeuristicNet, train_model  # Ensure model.py defines these correctly.
import Backgammon
import pubeval
import randomAgent


# ----- Use pubeval_eval instead of assess_board -----
def pubeval_eval(board,  min_val=0.0, max_val=100.0):
    """
    Evaluates a board using Tesauro's pubeval.
    Performs necessary board flipping, type conversion, and then calls the C library.
    """
    from pubeval import pubeval_flip, lib, intp, israce
    board_flipped = pubeval_flip(board.copy())
    board_flipped = board_flipped.astype(np.int32)
    race = israce(board)
    raw_value = lib.pubeval(race, board_flipped.ctypes.data_as(intp))
    raw_value = max(min_val, min(raw_value, max_val))
    normalized_value = (raw_value - min_val) / (max_val - min_val)
    return normalized_value

# ----- board_to_state returns 28 dimensions -----
def board_to_state(board):
    """
    Converts a Backgammon board (numpy array of length 29) into a 28-dimensional vector.
    - Positions 1 to 24 are normalized by dividing by 15.
    - Positions 25 to 28 are appended as extra features.
    Total dimension = 28.
    """
    positions = board[1:25].astype(np.float32) / 15.0  # 24 elements
    extras = board[25:29].astype(np.float32)  # 4 elements
    return np.concatenate((positions, extras))


# ----- Data Collection -----
def collect_data(n_boards):
    """
    Generates at least n_boards board states by playing random games (using randomAgent).
    Each board is labeled using the pubeval evaluation.
    Returns:
       states: numpy array of shape (n_boards, 28)
       labels: numpy array of shape (n_boards, 1)
    """
    boards = []
    labels = []
    while len(boards) < n_boards:
        board = Backgammon.init_board()
        # Play until game over or until we have enough boards.
        while not Backgammon.game_over(board) and len(boards) < n_boards:
            boards.append(board.copy())
            labels.append(pubeval_eval(board))
            dice = Backgammon.roll_dice()
            move = randomAgent.action(board, dice, 1, 0)
            if len(move) == 0:
                break
            # If the move is compound, update board sequentially.
            if hasattr(move, "ndim") and move.ndim == 2:
                for m in move:
                    board = Backgammon.update_board(board, m, 1)
            else:
                board = Backgammon.update_board(board, move, 1)
    states = np.array([board_to_state(b) for b in boards])
    labels = np.array(labels).reshape(-1, 1)
    print("Collected training data: states shape =", states.shape, "labels shape =", labels.shape)
    return states, labels


# ----- Network Evaluation on Test Set -----
def evaluate_network(network, test_states, test_labels):
    """
    Evaluates the network on the test set.
    Returns predictions and computes the mean and variance of absolute differences.
    """
    network.eval()
    predictions = []
    for state in test_states:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        predictions.append(network(state_tensor).item())
    differences = np.abs(np.array(predictions) - test_labels.flatten())
    mean_diff = np.mean(differences)
    var_diff = np.var(differences)
    return predictions, mean_diff, var_diff


# ----- Main Function -----
def main():
    # Instantiate a fresh (untrained) heuristic network with input dimension 28.
    network = HeuristicNet(input_size=28, hidden_size=40)
    print("Fresh heuristic network instantiated.")

    # Collect training data (2000 examples) using random play.
    print("Collecting training data (2000 examples) using pubeval heuristic...")
    train_states, train_labels = collect_data(2000)

    # Training parameters.
    num_epochs = 50
    batch_size = 32
    learning_rate = 0.001

    # Train the network to mimic the pubeval evaluation.
    print("Training network to mimic pubeval heuristic...")
    losses = train_model(network, train_states, train_labels,
                         num_epochs=num_epochs,
                         batch_size=batch_size,
                         learning_rate=learning_rate)

    # Plot training loss progression and save the plot.
    plt.figure()
    plt.plot(range(1, num_epochs + 1), losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss (MSE)")
    plt.title("Training Loss Progression")
    plt.savefig("training_loss_progress.png")
    print("Training loss plot saved as 'training_loss_progress.png'.")
    plt.show()

    # Collect a separate test set (200 examples).
    print("Collecting test data (200 examples)...")
    test_states, test_labels = collect_data(200)

    # Evaluate the network on the test set.
    predictions, mean_diff, var_diff = evaluate_network(network, test_states, test_labels)
    print("Evaluation on test data:")
    print("Mean absolute difference: {:.4f}".format(mean_diff))
    print("Variance of differences: {:.4f}".format(var_diff))

    # Create a scatter plot comparing network predictions vs. pubeval labels.
    plt.figure()
    plt.scatter(test_labels, predictions, alpha=0.7)
    plt.xlabel("Pubeval Evaluation (Label)")
    plt.ylabel("Network Prediction")
    plt.title("Network vs. Pubeval Evaluations (Fresh Network)")
    plt.savefig("network_vs_pubeval_test.png")
    print("Scatter plot saved as 'network_vs_pubeval_test.png'.")
    plt.show()

    # Save numerical results to a text file.
    with open("results.txt", "w") as f:
        f.write("Mean absolute difference: {:.4f}\n".format(mean_diff))
        f.write("Variance of differences: {:.4f}\n".format(var_diff))
    print("Results saved to 'results.txt'.")

    # Save the trained model.
    torch.save(network.state_dict(), "trained_network.pth")
    print("Final trained network saved as 'trained_network.pth'.")


if __name__ == '__main__':
    main()
