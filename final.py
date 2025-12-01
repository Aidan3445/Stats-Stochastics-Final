import marimo

__generated_with = "0.18.1"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    from typing import List
    from dataclasses import dataclass
    import sys

    @dataclass
    class Player:
        name: str
        service_win_p: float  # P(Win point | Serving)
        return_win_p: float   # P(Win point | Returning)

    def print_matrix(matrix: np.ndarray, states: List[str]):
        def parity(index: int) -> tuple:
            p = index % 2 == 0
            bg_color = "\x1b[100m" if p else "\x1b[40m"
            text_color = "\x1b[30m" if p else "\x1b[37m"
            return bg_color, text_color

        def colorize(value: float, index: int) -> str:
            formatted = f"{value:.3f}".rjust(5)
            bg_color, text_color = parity(index)

            if value == 0:
                return f"{bg_color}{text_color}{formatted}"
            elif value < 0.5:
                return f"{bg_color}\x1b[91m{formatted}{text_color}"
            elif value > 0.5:
                return f"{bg_color}\x1b[92m{formatted}{text_color}"
            else:
                return f"{bg_color}\x1b[93m{formatted}{text_color}"

        # Header
        header = "\x1b[1;40mState\x1b[0;40m|" + "|".join(state.rjust(5) for state in states)
        print(header)

        # Rows
        for i, row in enumerate(matrix):
            bg_color, text_color = parity(i)
            row_str = f"{bg_color}{text_color}{states[i].ljust(5)}|"
            row_str += "|".join(colorize(val, i) for val in row)
            row_str += "\x1b[0m"
            print(row_str)
    return Player, np, print_matrix


@app.cell
def _(Player, np, print_matrix):
    GAME_STATES = ["0-0", "15-0", "0-15", "15-15", "30-0", "0-30", "30-15", "15-30", "30-30", "40-0", "0-40", "40-15", "15-40", "40-30", "30-40", "Deuce", "AdIn", "AdOut", "Win", "Lose"]

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

            # Split matrix into transient (Q) and absorbing (R) parts
            # Last 2 states are absorbing (Win, Lose)
            Q = matrix[:-2, :-2]
            R = matrix[:-2, -2:]

            # Fundamental matrix: N = (I - Q)^-1
            I = np.identity(Q.shape[0])
            N = np.linalg.inv(I - Q)

            # Absorption probabilities: B = N * R
            B = N @ R

            # Return probability of winning from start_state (first column is Win)
            return B[state_index, 0]

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

    
    return (GameMatrix,)


@app.cell
def _(GameMatrix, Player, np, print_matrix):
    SET_STATES = [f"{a}-{b}" for a in range(6) for b in range(6)] + ["6-5", "5-6", "Win", "Lose"]

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

            # Create state mapping
            state_to_idx = {}
            idx = 0

            # Add all transient states (a, b) where a, b <= 5
            for a in range(6):
                for b in range(6):
                    state_to_idx[(a, b)] = idx
                    idx += 1

            # Add advantage states
            state_to_idx[(6, 5)] = idx  # Player ahead by 1
            idx += 1
            state_to_idx[(5, 6)] = idx  # Opponent ahead by 1
            idx += 1

            # Add absorbing states
            WIN_IDX = idx
            state_to_idx[("win",)] = WIN_IDX
            idx += 1
            LOSE_IDX = idx
            state_to_idx[("lose",)] = LOSE_IDX

            # Initialize matrix
            matrix = np.zeros((40, 40))

            # Fill in transitions
            for a in range(6):
                for b in range(6):
                    curr_idx = state_to_idx[(a, b)]

                    # Determine who is serving (alternates)
                    # Assuming player serves on even total games
                    total_games = a + b
                    player_serving = (total_games % 2 == 0)

                    if player_serving:
                        win_prob = w
                        lose_prob = l
                    else:
                        win_prob = rw
                        lose_prob = rl

                    # Determine next states
                    if a < 5:
                        # Player can win a game without reaching set point
                        next_win = state_to_idx[(a + 1, b)]
                    elif a == 5 and b < 5:
                        # Player at 5, opponent less than 5: winning game wins set
                        next_win = WIN_IDX
                    elif a == 5 and b == 5:
                        # At 5-5, winning goes to 6-5 (advantage)
                        next_win = state_to_idx[(6, 5)]

                    if b < 5:
                        # Opponent can win a game without reaching set point
                        next_lose = state_to_idx[(a, b + 1)]
                    elif b == 5 and a < 5:
                        # Opponent at 5, player less than 5: losing game loses set
                        next_lose = LOSE_IDX
                    elif a == 5 and b == 5:
                        # At 5-5, losing goes to 5-6 (disadvantage)
                        next_lose = state_to_idx[(5, 6)]

                    matrix[curr_idx, next_win] = win_prob
                    matrix[curr_idx, next_lose] = lose_prob

            # Handle advantage states
            # (6, 5): Player ahead by 1
            adv_player_idx = state_to_idx[(6, 5)]
            total_games = 11  # 6 + 5
            player_serving = (total_games % 2 == 0)

            if player_serving:
                win_prob = w
                lose_prob = l
            else:
                win_prob = rw
                lose_prob = rl

            matrix[adv_player_idx, WIN_IDX] = win_prob  # Win the set
            matrix[adv_player_idx, state_to_idx[(5, 5)]] = lose_prob  # Back to deuce

            # (5, 6): Opponent ahead by 1
            adv_opp_idx = state_to_idx[(5, 6)]
            total_games = 11  # 5 + 6
            player_serving = (total_games % 2 == 0)

            if player_serving:
                win_prob = w
                lose_prob = l
            else:
                win_prob = rw
                lose_prob = rl

            matrix[adv_opp_idx, state_to_idx[(5, 5)]] = win_prob  # Back to deuce
            matrix[adv_opp_idx, LOSE_IDX] = lose_prob  # Lose the set

            # Absorbing states
            matrix[WIN_IDX, WIN_IDX] = 1.0
            matrix[LOSE_IDX, LOSE_IDX] = 1.0

            return matrix

        def get_set_win_probability(self, start_state: str="0-0") -> float:
            state_index = SET_STATES.index(start_state)

            # Split matrix
            Q = self.matrix[:-2, :-2]
            R = self.matrix[:-2, -2:]

            # Fundamental matrix
            I = np.eye(Q.shape[0])
            N = np.linalg.inv(I - Q)

            # Absorption probabilities
            B = N @ R

            return B[state_index, 0]

        def print_set_matrix(self):
            print(f"Matrix for {self.player.name} (Set):")
            print_matrix(self.matrix, SET_STATES)
    return (SetMatrix,)


@app.cell
def _(Player, SetMatrix, np, print_matrix):
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
        
            # Split matrix into transient (Q) and absorbing (R) parts
            # Last 2 states are absorbing (Win, Lose)
            Q = self.matrix[:-2, :-2]
            R = self.matrix[:-2, -2:]

            # Fundamental matrix: N = (I - Q)^-1
            I = np.identity(Q.shape[0])
            N = np.linalg.inv(I - Q)

            # Absorption probabilities: B = N * R
            B = N @ R

            # Return probability of winning from start_state (first column is Win)
            return B[state_index, 0]

        def print_match_matrix(self):
            print(f"\nTransition Matrix for {self.player.name} (Match):")
            print_matrix(self.matrix, MATCH_STATES)
    return (MatchMatrix,)


@app.cell
def _(GameMatrix, MatchMatrix, Player, SetMatrix):
    sab = Player(
        name="Sabalenka",
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
    return sabalenka_game, set_matrix


@app.cell
def _(sabalenka_game, set_matrix):
    sabalenka_game.print_server_matrix()
    print("~" * 40)
    sabalenka_game.print_returner_matrix()
    print("~" * 40)
    set_matrix.print_set_matrix()
    return


@app.cell
def _():
    x = "https://www.atptour.com/en/players/nick-kyrgios/ke17/player-stats?year=all&surface=all"

    x.split("/")
    return


if __name__ == "__main__":
    app.run()
