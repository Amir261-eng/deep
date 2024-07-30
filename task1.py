import random

def print_board(board):
    """Prints the Tic-Tac-Toe board."""
    print("-------------")
    for row in board:
        print("|", " | ".join(row), "|")
        print("-------------")

def check_winner(board):
    """Checks if there is a winner or if the board is full."""
    # Check rows and columns
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != ' ':
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != ' ':
            return board[0][i]

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] != ' ':
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != ' ':
        return board[0][2]
    
    # Check for tie
    if all(board[row][col] != ' ' for row in range(3) for col in range(3)):
        return 'tie'

    return None

def evaluate(board):
    """Evaluates the current position for the minimax algorithm."""
    winner = check_winner(board)
    if winner == 'X':
        return 1
    elif winner == 'O':
        return -1
    return 0  # Tie

def minimax(board, depth, is_maximizing):
    """Minimax algorithm implementation."""
    result = check_winner(board)
    if result is not None:
        return evaluate(board)

    if is_maximizing:
        best_value = -float('inf')
        for row in range(3):
            for col in range(3):
                if board[row][col] == ' ':
                    board[row][col] = 'X'
                    value = minimax(board, depth + 1, False)
                    board[row][col] = ' '
                    best_value = max(best_value, value)
        return best_value
    else:
        best_value = float('inf')
        for row in range(3):
            for col in range(3):
                if board[row][col] == ' ':
                    board[row][col] = 'O'
                    value = minimax(board, depth + 1, True)
                    board[row][col] = ' '
                    best_value = min(best_value, value)
        return best_value

def get_best_move(board):
    """Gets the best move for the computer using minimax."""
    best_move = None
    best_value = -float('inf')
    for row in range(3):
        for col in range(3):
            if board[row][col] == ' ':
                board[row][col] = 'X'
                move_value = minimax(board, 0, False)
                board[row][col] = ' '
                if move_value > best_value:
                    best_value = move_value
                    best_move = (row, col)
    return best_move

def main():
    board = [[' ' for _ in range(3)] for _ in range(3)]
    print("Welcome to Tic-Tac-Toe!")
    print_board(board)

    while True:
        while True:
            try:
                row = int(input("Enter the row (0, 1, 2): "))
                col = int(input("Enter the column (0, 1, 2): "))
                if 0 <= row <= 2 and 0 <= col <= 2:
                    if board[row][col] == ' ':
                        board[row][col] = 'O'
                        break
                    else:
                        print("That spot is taken! Try again.")
                else:
                    print("Invalid input. Row and column must be 0, 1, or 2.")
            except ValueError:
                print("Invalid input. Please enter integers for row and column.")

        print_board(board)
        winner = check_winner(board)
        if winner:
            if winner == 'tie':
                print("It's a tie!")
            else:
                print(f"Congratulations! {winner} wins!")
            break
        
        print("Computer's turn...")
        row, col = get_best_move(board)
        board[row][col] = 'X'
        
        print_board(board)
        winner = check_winner(board)
        if winner:
            if winner == 'tie':
                print("It's a tie!")
            else:
                print("Computer wins!")
            break

if __name__ == "__main__":
    main()
