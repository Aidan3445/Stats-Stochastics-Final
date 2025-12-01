import numpy as np
from typing import List
from dataclasses import dataclass
import sys
import requests

@dataclass
class Player:
    name: str
    service_win_p: float  # P(Win point | Serving)
    return_win_p: float   # P(Win point | Returning)

def print_matrix(matrix: np.ndarray, states: List[str]):
    def parity(index: int) -> tuple:
        p = index % 2 == 0
        bg_color = "\x1b[100m" if p else "\x1b[40m"  # White/Black background
        text_color = "\x1b[30m" if p else "\x1b[37m" # Black/White text
        return bg_color, text_color

    def colorize(value: float, index: int) -> str:
        formatted = f"{value:.3f}".rjust(5)
        bg_color, text_color = parity(index)

        if value == 0:
            return f"{bg_color}{text_color}{formatted}"
        elif value < 0.5:
            return f"{bg_color}\x1b[91m{formatted}{text_color}" # Red text
        elif value > 0.5:
            return f"{bg_color}\x1b[92m{formatted}{text_color}" # Green text
        else:
            return f"{bg_color}\x1b[93m{formatted}{text_color}" # Yellow text

    # Header
    header = "\x1b[1;40mState\x1b[0;40m|" + "|".join(state.rjust(5) for state in states)
    print(header)

    # Rows
    for i, row in enumerate(matrix):
        bg_color, text_color = parity(i)
        row_str = f"{bg_color}{text_color}{states[i].ljust(5)}|"
        row_str += "|".join(colorize(val, i) for val in row)
        row_str += "\x1b[0m" # Reset colors

        print(row_str)


GAME_STATES = ["0-0", "15-0", "0-15", "15-15", "30-0", "0-30", "30-15", "15-30", "30-30", "40-0", 
               "0-40", "40-15", "15-40", "40-30", "30-40", "Deuce", "AdIn", "AdOut", "Win", "Lose"]

class GameMatrix:
    def __init__(self, player: Player):
        self.player = player
        self.matrix_server = self._build_server_matrix(player.service_win_p)
        self.matrix_returner = self._build_server_matrix(player.return_win_p)

    @staticmethod
    def _build_server_matrix(w: float) -> np.ndarray:
        l = 1 - w

        matrix = np.array([
            [0, w, l, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0-0
            [0, 0, 0, l, w, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 15-0
            [0, 0, 0, w, 0, l, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0-15
            [0, 0, 0, 0, 0, 0, w, l, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 15-15
            [0, 0, 0, 0, 0, 0, l, 0, 0, w, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 30-0
            [0, 0, 0, 0, 0, 0, 0, w, 0, 0, l, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0-30
            [0, 0, 0, 0, 0, 0, 0, 0, l, 0, 0, w, 0, 0, 0, 0, 0, 0, 0, 0],  # 30-15
            [0, 0, 0, 0, 0, 0, 0, 0, w, 0, 0, 0, l, 0, 0, 0, 0, 0, 0, 0],  # 15-30
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, w, l, 0, 0, 0, 0, 0],  # 30-30
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, l, 0, 0, 0, 0, 0, 0, w, 0],  # 40-0
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, w, 0, 0, 0, 0, 0, 0, l],  # 0-40
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, l, 0, 0, 0, 0, w, 0],  # 40-15
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, w, 0, 0, 0, 0, l],  # 15-40
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, l, 0, 0, w, 0],  # 40-30
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, w, 0, 0, 0, l],  # 30-40
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, w, l, 0, 0],  # Deuce
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, l, 0, 0, w, 0],  # Ad-In
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, w, 0, 0, 0, l],  # Ad-Out
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Game-W
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Game-L
        ])

        return matrix

    @staticmethod
    def _get_win_probability(matrix: np.ndarray, start_state: str) -> float:
        state_index = GAME_STATES.index(start_state)

        Q = matrix[:-2, :-2]
        R = matrix[:-2, -2:]

        I = np.identity(Q.shape[0])
        N = np.linalg.inv(I - Q)

        probs = N @ R

        return probs[state_index, 0]

    def get_server_win_probability(self, start_state: str="0-0") -> float:
        return self._get_win_probability(self.matrix_server, start_state)

    def get_returner_win_probability(self, start_state: str="0-0") -> float:
        return self._get_win_probability(self.matrix_returner, start_state)

    def print_server_matrix(self):
        print(f"Matrix for {self.player.name} (Serving):")
        print_matrix(self.matrix_server, GAME_STATES)

    def print_returner_matrix(self):
        print(f"Matrix for {self.player.name} (Returning):")
        print_matrix(self.matrix_returner, GAME_STATES)



SET_STATES = [f"{a}-{b}" for b in range(6) for a in range(6)] + ["6-5", "5-6", "Win", "Lose"]

class SetMatrix:
    def __init__(self, player: Player):
        self.player = player
        game_matrix = GameMatrix(player)
        self.service_win_p = game_matrix.get_server_win_probability()
        self.return_win_p = game_matrix.get_returner_win_probability()
        self.matrix = self.build_set_matrix()

    def build_set_matrix(self) -> np.ndarray:
        w = self.service_win_p
        l = 1 - w
        rw = self.return_win_p
        rl = 1 - rw

        """ No tie break, win by two games. 
            A lost game at 6-5, for example, goes back to 5-5 (Deuce-like)
        States are organized as:
        - Transient states: all (a, b) where a, b in [0, 5] and not both >=5
        - Advantage states: (6, 5) and (5, 6)
        - Absorbing states: Win, Lose
        Total: 36 + 2 + 2 = 40 states
        """

        matrix = np.zeros((40, 40))

        for b in range(6):
            for a in range(6):
                index = b * 6 + a

                player_serving = (a + b) % 2 == 0

                if player_serving:
                    win_prob = w
                    lose_prob = l
                else:
                    win_prob = rw
                    lose_prob = rl

                if a < 5:
                    next_win = index + 1
                elif a == 5 and b < 5:
                    next_win = 38 # Win state index
                elif a == 5 and b == 5:
                    next_win = 36 # Advantage in state


                if b < 5:
                    next_lose = index + 6
                elif b == 5 and a < 5:
                    next_lose = 39 # Lose state index
                elif a == 5 and b == 5:
                    next_lose = 37 # Advantage out state

                matrix[index, next_win] = win_prob
                matrix[index, next_lose] = lose_prob

        # Simplified version we always serve the first game of the set ==> 6-5/5-6 we return
        win_prob = rw
        lose_prob = rl

        # Advantage in
        matrix[36, 38] = win_prob   # Win the set
        matrix[36, 35] = lose_prob  # Back to tie

        # Advantage out
        matrix[37, 35] = win_prob   # Back to tie
        matrix[37, 39] = lose_prob  # Lose the set

        # Win/Lose
        matrix[38, 38] = 1.0
        matrix[39, 39] = 1.0

        return matrix

    def get_set_win_probability(self, start_state: str="0-0") -> float:
        state_index = SET_STATES.index(start_state)

        Q = self.matrix[:-2, :-2]
        R = self.matrix[:-2, -2:]

        I = np.eye(Q.shape[0])
        N = np.linalg.inv(I - Q)

        probs = N @ R

        return probs[state_index, 0]

    def print_set_matrix(self):
        print(f"Matrix for {self.player.name} (Set):")
        print_matrix(self.matrix, SET_STATES)


MATCH_STATES = ["0-0", "1-0", "0-1", "1-1", "Win", "Lose"]

class MatchMatrix:
    def __init__(self, player: Player):
        self.player = player
        self.set_win_p = SetMatrix(player).get_set_win_probability()
        self.matrix = self.build_match_matrix()

    def build_match_matrix(self) -> np.ndarray:
        w = self.set_win_p
        l = 1 - w

        matrix = np.array([
            [0, w, l, 0, 0, 0], # 0-0
            [0, 0, 0, l, w, 0], # 1-0
            [0, 0, 0, w, 0, l], # 0-1
            [0, 0, 0, 0, w, l], # 1-1
            [0, 0, 0, 0, 1, 0], # Win
            [0, 0, 0, 0, 0, 1], # Lose
        ])

        return matrix

    def get_match_win_probability(self, start_state="0-0") -> float:
        state_index = MATCH_STATES.index(start_state)

        Q = self.matrix[:-2, :-2]
        R = self.matrix[:-2, -2:]

        I = np.identity(Q.shape[0])
        N = np.linalg.inv(I - Q)

        probs = N @ R

        return probs[state_index, 0]

    def print_match_matrix(self):
        print(f"Transition Matrix for {self.player.name} (Match):")
        print_matrix(self.matrix, MATCH_STATES)

sab = Player(
    name="Aryna Sabalenka",
    service_win_p = 0.599,
    return_win_p = 0.448
)

# Sabalenka
print(f"\nPlayer: {sab.name}")
print(f"Service Point Win Probability: {sab.service_win_p}")
print(f"Return Point Win Probability: {sab.return_win_p}")
sabalenka_game = GameMatrix(sab)
print(f"Probability of winning a game when serving: "
      f"{sabalenka_game.get_server_win_probability():.3f}")
print(f"Probability of winning a game when returning: "
      f"{sabalenka_game.get_returner_win_probability():.3f}")

set_matrix = SetMatrix(sab)
print(f"Probability of winning a set: "
      f"{set_matrix.get_set_win_probability():.3f}")

match_matrix = MatchMatrix(sab)
print(f"Probability of winning a match: "
      f"{match_matrix.get_match_win_probability():.3f}")

sabalenka_game.print_server_matrix()
print("~" * 40)
sabalenka_game.print_returner_matrix()
print("~" * 40)
set_matrix.print_set_matrix()

# input data
while True:
    print("\nDo you want to analyze another player? (y/n): ", end="")
    choice = input().strip().lower()
    if choice == 'n':
        sys.exit(0)
    elif choice != 'y':
        print("Invalid choice. Please enter 'y' or 'n'.")
        continue

    print("Enter atptour.com or wtatennis.com player stats URL: \
(e.g https://www.atptour.com/en/players/nick-kyrgios/ke17/player-stats)\n\
or press Enter to manually input data: ", end="")

    url = input().strip()
    if url:
        is_atp = "atptour.com" in url
        is_wta = "wtatennis.com" in url
        if is_atp:
            id = url.split("/")[6]
            stats_url = f"https://atptour.com/en/-/www/stats/{id}/2025/all?v=1"
        if is_wta:
            id = url.split("/")[4]
            stats_url = f"https://api.wtatennis.com/tennis/players/{id}/year/2025"

    try:
        response = requests.get(stats_url)
        response.raise_for_status()
        data = response.json()
        if is_atp:
            service_win_p = data["Stats"]["ServiceRecordStats"]["ServicePointsWonPercentage"] / 100.0
            return_win_p = data["Stats"]["ReturnRecordStats"]["ReturnPointsWonPercentage"] / 100.0
            name = url.split("/")[5].replace("-", " ").title()
        if is_wta:
            service_win_p = data["stats"]["service_points_won_percent"] / 100.0
            return_win_p = data["stats"]["return_points_won_percent"] / 100.0
            name = data["player"]["fullName"]

        player = Player(
            name=name,
            service_win_p=service_win_p,
            return_win_p=return_win_p
        )
    except Exception as e:
        print(f"Error fetching or parsing data: {e}")
        print("Player not found, enter data manually.")
        print("Enter player name: ", end="")
        name = input().strip()
        print("Enter service point win probability (e.g., 0.65): ", end="")
        service_win_p = float(input().strip())
        print("Enter return point win probability (e.g., 0.45): ", end="")
        return_win_p = float(input().strip())

        player = Player(
            name=name,
            service_win_p=service_win_p,
            return_win_p=return_win_p
        )

    print(f"\nPlayer: {player.name}")
    print(f"Service Point Win Probability: {player.service_win_p:.3f}")
    print(f"Return Point Win Probability: {player.return_win_p:.3f}")
    game_matrix = GameMatrix(player)
    print(f"Probability of winning a game when serving: "
          f"{game_matrix.get_server_win_probability():.3f}")
    print(f"Probability of winning a game when returning: "
          f"{game_matrix.get_returner_win_probability():.3f}")
    set_matrix = SetMatrix(player)
    print(f"Probability of winning a set: "
            f"{set_matrix.get_set_win_probability():.3f}")
    match_matrix = MatchMatrix(player)
    print(f"Probability of winning a match: "
          f"{match_matrix.get_match_win_probability():.3f}")

    print("\nDo you want to see the transition matrices? (y/n): ", end="")
    show_matrices = input().strip().lower()
    if show_matrices == 'y':
        game_matrix.print_server_matrix()
        print("~" * 40)
        game_matrix.print_returner_matrix()
        print("~" * 40)
        set_matrix.print_set_matrix()
        print("~" * 40)
        match_matrix.print_match_matrix()
