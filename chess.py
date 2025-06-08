import pygame
import sys
import time
# Initialize pygame
pygame.init()

# Constants
SIDEBAR_WIDTH = 200
BOARD_SIZE = 800
WIDTH, HEIGHT = SIDEBAR_WIDTH + BOARD_SIZE, BOARD_SIZE
ROWS, COLS = 8, 8
SQUARE_SIZE = BOARD_SIZE // COLS

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
SIDEBAR_BG = (200, 200, 200)
HIGHLIGHT = (186, 202, 68)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Fonts
FONT = pygame.font.SysFont('Arial', 24)
SMALL_FONT = pygame.font.SysFont('Arial', 18)
LARGE_FONT = pygame.font.SysFont('Arial', 36)

#--------------------------------------------------------------
# Piece-value table for evaluation function
piece_values = {
    'p': 1,
    'N': 3,
    'B': 3,
    'R': 5,
    'Q': 9,
    'K': 0  # King's value is handled by checkmate detection
}

def evaluate_board(board):
   
    score = 0
    for row in board:
        for cell in row:
            if cell != " ":
                color = 1 if cell[0] == 'w' else -1
                score += color * piece_values.get(cell[1], 0)
    return score

def minimax(board, depth, alpha, beta, maximizing_player):
   
    current_color = 'w' if maximizing_player else 'b'

    # Terminal checks: checkmate or stalemate
    if depth == 0:
        return evaluate_board(board), None

    if is_in_check(board, current_color) and not has_legal_moves(board, current_color):
        # Checkmate: if White is checkmated at maximizing node → large negative score
        return (-9999 if maximizing_player else 9999), None

    if not has_legal_moves(board, current_color):
        # Stalemate
        return 0, None

    best_move = None
    if maximizing_player:
        max_eval = -float('inf')
        for r in range(ROWS):
            for c in range(COLS):
                if board[r][c] != " " and board[r][c][0] == 'w':
                    candidate_moves = get_valid_moves(board, r, c)
                    for mv in candidate_moves:
                        new_board = make_move(board, (r, c), mv)
                        if is_in_check(new_board, 'w'):
                            continue
                        eval_score, _ = minimax(new_board, depth - 1, alpha, beta, False)
                        if eval_score > max_eval:
                            max_eval = eval_score
                            best_move = ((r, c), mv)
                        alpha = max(alpha, eval_score)
                        if beta <= alpha:
                            break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for r in range(ROWS):
            for c in range(COLS):
                if board[r][c] != " " and board[r][c][0] == 'b':
                    candidate_moves = get_valid_moves(board, r, c)
                    for mv in candidate_moves:
                        new_board = make_move(board, (r, c), mv)
                        if is_in_check(new_board, 'b'):
                            continue
                        eval_score, _ = minimax(new_board, depth - 1, alpha, beta, True)
                        if eval_score < min_eval:
                            min_eval = eval_score
                            best_move = ((r, c), mv)
                        beta = min(beta, eval_score)
                        if beta <= alpha:
                            break
        return min_eval, best_move

#--------------------------------------------------------------
def draw_board():
    for row in range(ROWS):
        for col in range(COLS):
            color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
            pygame.draw.rect(
                screen,
                color,
                (SIDEBAR_WIDTH + col * SQUARE_SIZE,
                 row * SQUARE_SIZE,
                 SQUARE_SIZE,
                 SQUARE_SIZE)
            )

def load_images():
    pieces = ["wp", "wR", "wN", "wB", "wQ", "wK",
              "bp", "bR", "bN", "bB", "bQ", "bK"]
    images = {}
    for piece in pieces:
        images[piece] = pygame.transform.scale(
            pygame.image.load(f"images/{piece}.png"),
            (SQUARE_SIZE, SQUARE_SIZE)
        )
    return images

def draw_pieces(board, images):
    for row in range(ROWS):
        for col in range(COLS):
            piece = board[row][col]
            if piece != " ":
                screen.blit(
                    images[piece],
                    (SIDEBAR_WIDTH + col * SQUARE_SIZE,
                     row * SQUARE_SIZE)
                )

def highlight_squares(squares):
    for (row, col) in squares:
        pygame.draw.rect(
            screen,
            HIGHLIGHT,
            (SIDEBAR_WIDTH + col * SQUARE_SIZE,
             row * SQUARE_SIZE,
             SQUARE_SIZE,
             SQUARE_SIZE),
            5
        )

def create_board():
    return [
        ["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"],
        ["bp", "bp", "bp", "bp", "bp", "bp", "bp", "bp"],
        [" ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " "],
        ["wp", "wp", "wp", "wp", "wp", "wp", "wp", "wp"],
        ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"]
    ]

def draw_text_center(surface, text, pos, font, color):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=pos)
    surface.blit(text_surface, text_rect)

def draw_sidebar(w_time, b_time, captured_white, captured_black, images):
    pygame.draw.rect(screen, SIDEBAR_BG, (0, 0, SIDEBAR_WIDTH, HEIGHT))
    draw_text_center(screen, "White Time", (SIDEBAR_WIDTH // 2, 30), FONT, BLACK)
    draw_text_center(screen, format_time(w_time), (SIDEBAR_WIDTH // 2, 60), FONT, BLACK)
    draw_text_center(screen, "Black Time", (SIDEBAR_WIDTH // 2, HEIGHT // 2 + 30), FONT, BLACK)
    draw_text_center(screen, format_time(b_time), (SIDEBAR_WIDTH // 2, HEIGHT // 2 + 60), FONT, BLACK)
    draw_text_center(screen, "Captured by Black", (SIDEBAR_WIDTH // 2, 100), SMALL_FONT, BLACK)
    draw_captured_pieces(captured_white, images, 120)
    draw_text_center(screen, "Captured by White", (SIDEBAR_WIDTH // 2, HEIGHT // 2 + 100), SMALL_FONT, BLACK)
    draw_captured_pieces(captured_black, images, HEIGHT // 2 + 120)

def draw_captured_pieces(pieces_list, images, start_y):
    x = 10
    y = start_y
    size = SQUARE_SIZE // 2
    for piece in pieces_list:
        img = pygame.transform.scale(images[piece], (size, size))
        screen.blit(img, (x, y))
        x += size + 5
        if x + size > SIDEBAR_WIDTH:
            x = 10
            y += size + 5

def format_time(seconds):
    seconds = max(0, int(seconds))
    m = seconds // 60
    s = seconds % 60
    return f"{m:02d}:{s:02d}"

def time_selection_screen():
    input_time = 5
    input_active = True
    input_str = ""
    while input_active:
        screen.fill(GREEN)
        draw_text_center(screen, "You wanna play against me(the unbeatable) then here I am! Enter time per player (minutes):",
                         (WIDTH//2, HEIGHT//2 - 40), FONT, BLACK)
        draw_text_center(screen, input_str + "|",
                         (WIDTH//2, HEIGHT//2), FONT, BLACK)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    if input_str.isdigit() and int(input_str) > 0:
                        input_time = int(input_str)
                        input_active = False
                elif event.key == pygame.K_BACKSPACE:
                    input_str = input_str[:-1]
                else:
                    if event.unicode.isdigit():
                        input_str += event.unicode
    return input_time * 60

def get_valid_moves(board, row, col):
    piece = board[row][col]
    if piece == " ":
        return []
    moves = []
    color = piece[0]
    kind = piece[1]
    directions = []

    if kind == "p":
        dir = -1 if color == "w" else 1
        start_row = 6 if color == "w" else 1
        # Move forward 1
        if 0 <= row + dir < ROWS and board[row + dir][col] == " ":
            moves.append((row + dir, col))
            # Move forward 2 on first move
            if row == start_row and board[row + 2 * dir][col] == " ":
                moves.append((row + 2 * dir, col))
        # Capture diagonally
        for dc in [-1, 1]:
            r, c = row + dir, col + dc
            if 0 <= r < ROWS and 0 <= c < COLS:
                if board[r][c] != " " and board[r][c][0] != color:
                    moves.append((r, c))

    elif kind == "R":
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    elif kind == "B":
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    elif kind == "Q":
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]

    elif kind == "N":
        knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                        (1, -2), (1, 2), (2, -1), (2, 1)]
        for dr, dc in knight_moves:
            r, c = row + dr, col + dc
            if 0 <= r < ROWS and 0 <= c < COLS:
                if board[r][c] == " " or board[r][c][0] != color:
                    moves.append((r, c))

    elif kind == "K":
        king_moves = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1), (0, 1),
                      (1, -1), (1, 0), (1, 1)]
        for dr, dc in king_moves:
            r, c = row + dr, col + dc
            if 0 <= r < ROWS and 0 <= c < COLS:
                if board[r][c] == " " or board[r][c][0] != color:
                    moves.append((r, c))

    # For sliding pieces (R, B, Q)
    for dr, dc in directions:
        r, c = row + dr, col + dc
        while 0 <= r < ROWS and 0 <= c < COLS:
            if board[r][c] == " ":
                moves.append((r, c))
            elif board[r][c][0] != color:
                moves.append((r, c))
                break
            else:
                break
            r += dr
            c += dc

    return moves

def find_king(board, color):
    for r in range(ROWS):
        for c in range(COLS):
            if board[r][c] == color + "K":
                return (r, c)
    return None

def is_square_attacked(board, row, col, attacker_color):
    # Check if square (row, col) is attacked by any piece of attacker_color

    # Directions for R, B, Q
    directions = {
        'R': [(-1, 0), (1, 0), (0, -1), (0, 1)],
        'B': [(-1, -1), (-1, 1), (1, -1), (1, 1)],
        'Q': [(-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, -1), (-1, 1), (1, -1), (1, 1)]
    }

    # Check for pawn attacks
    pawn_dir = -1 if attacker_color == 'w' else 1
    pawn_attack_squares = [(row + pawn_dir, col - 1), (row + pawn_dir, col + 1)]
    for r, c in pawn_attack_squares:
        if 0 <= r < ROWS and 0 <= c < COLS:
            if board[r][c] == attacker_color + "p":
                return True

    # Check knight attacks
    knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                    (1, -2), (1, 2), (2, -1), (2, 1)]
    for dr, dc in knight_moves:
        r, c = row + dr, col + dc
        if 0 <= r < ROWS and 0 <= c < COLS:
            if board[r][c] == attacker_color + "N":
                return True

    # Check king attacks (adjacent squares)
    king_moves = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1), (0, 1),
                  (1, -1), (1, 0), (1, 1)]
    for dr, dc in king_moves:
        r, c = row + dr, col + dc
        if 0 <= r < ROWS and 0 <= c < COLS:
            if board[r][c] == attacker_color + "K":
                return True

    # Check sliding pieces: rook, bishop, queen
    for piece_type, dirs in directions.items():
        for dr, dc in dirs:
            r, c = row + dr, col + dc
            while 0 <= r < ROWS and 0 <= c < COLS:
                current = board[r][c]
                if current != " ":
                    if current[0] == attacker_color and current[1] == piece_type:
                        return True
                    else:
                        break
                r += dr
                c += dc

    return False

def is_in_check(board, color):
    king_pos = find_king(board, color)
    if king_pos is None:
        return True  # No king => treated as check
    return is_square_attacked(board, king_pos[0], king_pos[1], 'w' if color == 'b' else 'b')

def make_move(board, from_pos, to_pos):
    new_board = [row[:] for row in board]
    piece = new_board[from_pos[0]][from_pos[1]]
    new_board[to_pos[0]][to_pos[1]] = piece
    new_board[from_pos[0]][from_pos[1]] = " "
    return new_board

def has_legal_moves(board, color):
    for r in range(ROWS):
        for c in range(COLS):
            if board[r][c] != " " and board[r][c][0] == color:
                candidate_moves = get_valid_moves(board, r, c)
                for mv in candidate_moves:
                    test_board = make_move(board, (r, c), mv)
                    if not is_in_check(test_board, color):
                        return True
    return False

def promote_pawn(board, row, col, color):
    # Auto promote to queen
    board[row][col] = color + "Q"

#--------------------------------------------------------------
def ai_move(board, color):
    """
    Use minimax to pick a move for 'color' ('b' for Black). 
    Returns (new_board, from_pos, to_pos, captured_piece).
    """
    _, best = minimax(board, depth=3, alpha=-float('inf'), beta=float('inf'),
                      maximizing_player=(color == 'w'))
    if not best:
        return board, None, None, None

    (r_from, c_from), (r_to, c_to) = best
    captured = board[r_to][c_to]
    new_board = make_move(board, (r_from, c_from), (r_to, c_to))
    piece_moved = new_board[r_to][c_to]
    # Pawn promotion for AI
    if piece_moved[1] == 'p' and (r_to == 0 or r_to == 7):
        promote_pawn(new_board, r_to, c_to, piece_moved[0])
    return new_board, (r_from, c_from), (r_to, c_to), captured

#--------------------------------------------------------------
# NEW FUNCTION FOR PLAY AGAIN FEATURE
def draw_button(surface, text, pos, size, color, hover_color, text_color=BLACK):
    mouse = pygame.mouse.get_pos()
    clicked = pygame.mouse.get_pressed()[0]
    
    rect = pygame.Rect(pos[0], pos[1], size[0], size[1])
    
    if rect.collidepoint(mouse):
        pygame.draw.rect(surface, hover_color, rect)
        if clicked:
            return True
    else:
        pygame.draw.rect(surface, color, rect)
    
    text_surf = FONT.render(text, True, text_color)
    text_rect = text_surf.get_rect(center=rect.center)
    surface.blit(text_surf, text_rect)
    return False

def show_game_over_screen(winner):
    # Dark overlay
    s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    s.fill((0, 0, 0, 180))  # Semi-transparent black
    screen.blit(s, (0, 0))
    
    draw_text_center(
        screen,
        f"Game Over: {winner}",
        (WIDTH // 2, HEIGHT // 2 - 50),
        LARGE_FONT, WHITE
    )
    
    # Play Again button
    play_again = draw_button(
        screen, "Play Again", 
        (WIDTH // 2 - 100, HEIGHT // 2 + 20), 
        (200, 50), GREEN, (100, 255, 100), WHITE
    )
    
    # Quit button
    quit_game = draw_button(
        screen, "Quit", 
        (WIDTH // 2 - 100, HEIGHT // 2 + 90), 
        (200, 50), RED, (255, 100, 100), WHITE
    )
    
    pygame.display.flip()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if (WIDTH // 2 - 100) <= mouse_pos[0] <= (WIDTH // 2 + 100):
                    if (HEIGHT // 2 + 20) <= mouse_pos[1] <= (HEIGHT // 2 + 70):
                        return True  # Play Again
                    elif (HEIGHT // 2 + 90) <= mouse_pos[1] <= (HEIGHT // 2 + 140):
                        return False  # Quit
        clock.tick(60)

# Set up display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Python Chess")
clock = pygame.time.Clock()

def main():
    while True:  # Outer loop for game restarts
        # Initialize game state
        board = create_board()
        images = load_images()
        selected_piece = None
        valid_moves = []
        turn = "w"
        w_time = time_selection_screen()
        b_time = w_time
        w_captured = []
        b_captured = []
        last_time = time.time()
        game_over = False
        winner = None
        message = ""

        # Main game loop
        running = True
        while running:
            current_time = time.time()
            elapsed = current_time - last_time
            last_time = current_time

            if not game_over:
                # Decrease timer
                if turn == "w":
                    w_time -= elapsed
                    if w_time <= 0:
                        game_over = True
                        winner = "Black wins on time!"
                else:
                    b_time -= elapsed
                    if b_time <= 0:
                        game_over = True
                        winner = "White wins on time!"

            # Handle user events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if game_over:
                    continue

                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = event.pos
                    if mouse_x >= SIDEBAR_WIDTH and mouse_y < BOARD_SIZE:
                        col = (mouse_x - SIDEBAR_WIDTH) // SQUARE_SIZE
                        row = mouse_y // SQUARE_SIZE
                        if selected_piece:
                            # Try to move to clicked square if valid
                            if (row, col) in valid_moves:
                                piece_moved = board[selected_piece[0]][selected_piece[1]]
                                captured = board[row][col]
                                board = make_move(board, selected_piece, (row, col))

                                # Capture logic
                                if captured != " ":
                                    if captured[0] == "w":
                                        w_captured.append(captured)
                                    else:
                                        b_captured.append(captured)

                                # Pawn promotion (human)
                                if piece_moved[1] == 'p' and (row == 0 or row == 7):
                                    promote_pawn(board, row, col, piece_moved[0])

                                # Illegal move check (king in check)
                                if is_in_check(board, turn):
                                    # Undo move
                                    board = make_move(board, (row, col), selected_piece)
                                    board[row][col] = captured
                                    message = "Illegal move: King would be in check!"
                                else:
                                    turn = "b"
                                    message = ""
                                selected_piece = None
                                valid_moves = []
                            else:
                                # If clicked own piece, reselect; otherwise deselect
                                if board[row][col] != " " and board[row][col][0] == turn:
                                    selected_piece = (row, col)
                                    candidate_moves = get_valid_moves(board, row, col)
                                    valid_moves = []
                                    for mv in candidate_moves:
                                        test_board = make_move(board, (row, col), mv)
                                        if not is_in_check(test_board, turn):
                                            valid_moves.append(mv)
                                else:
                                    selected_piece = None
                                    valid_moves = []
                        else:
                            # Select a piece of current player
                            if board[row][col] != " " and board[row][col][0] == turn:
                                selected_piece = (row, col)
                                candidate_moves = get_valid_moves(board, row, col)
                                valid_moves = []
                                for mv in candidate_moves:
                                    test_board = make_move(board, (row, col), mv)
                                    if not is_in_check(test_board, turn):
                                        valid_moves.append(mv)

            # After human moves (turn might be 'b'), let AI move
            if turn == 'b' and not game_over:
                # AI chooses and executes its move
                new_board, from_pos, to_pos, captured = ai_move(board, 'b')
                if from_pos and to_pos:
                    board = new_board
                    # AI capture
                    if captured != " ":
                        if captured[0] == 'w':
                            w_captured.append(captured)
                        else:
                            b_captured.append(captured)
                    turn = 'w'

            # Check for checkmate or stalemate
            if not game_over:
                if is_in_check(board, turn) and not has_legal_moves(board, turn):
                    game_over = True
                    winner = ("Black wins by checkmate!" if turn == "w" else "White wins by checkmate!")
                elif not is_in_check(board, turn) and not has_legal_moves(board, turn):
                    game_over = True
                    winner = "Stalemate!"

            # Draw everything
            screen.fill(WHITE)
            draw_board()
            if selected_piece:
                row, col = selected_piece
                pygame.draw.rect(
                    screen, RED,
                    (SIDEBAR_WIDTH + col * SQUARE_SIZE,
                     row * SQUARE_SIZE,
                     SQUARE_SIZE, SQUARE_SIZE),
                    4
                )
                highlight_squares(valid_moves)
            draw_pieces(board, images)
            draw_sidebar(w_time, b_time, w_captured, b_captured, images)
            if message:
                draw_text_center(
                    screen,
                    message,
                    (SIDEBAR_WIDTH + BOARD_SIZE // 2, BOARD_SIZE + 20),
                    FONT, RED
                )
            
            if game_over:
                if show_game_over_screen(winner):
                    break  # Break inner loop to restart game
                else:
                    pygame.quit()
                    sys.exit()

            pygame.display.flip()
            clock.tick(60)

if __name__ == "__main__":
    main()
