import numpy as np
from typing import List
from dataclasses import dataclass

GAME_STATES = [
    'Lv-Lv', '15-Lv', 'Lv-15', '15-15', '30-Lv', 'Lv-30', '30-15', '15-30', '30-30', '40-Lv',
    'Lv-40', '40-15', '15-40', '40-30', '30-40', 'Deuce', 'AdIn', 'AdOut', 'Win', 'Lose'
]

@dataclass
class Player:
    name: str
    service_win_p: float  # P(Win point | Serving)
    return_win_p: float   # P(Win point | Returning)


class GameMatrix:
    def __init__(self, player: Player):
        self.player = player
        self.matrix_server = self._build_server_matrix(player.service_win_p)
        self.matrix_returner = self._build_server_matrix(player.return_win_p)
    
    @staticmethod
    def _build_server_matrix(w: float) -> np.ndarray:
        """Build transition matrix for a given win probability."""
        l = 1 - w
        
        # States: 0-0|15-0|0-15|15-15|30-0|0-30|30-15|15-30|30-30|40-0|0-40|40-15|15-40|40-30|30-40|Deuce|Ad-In|Ad-Out|Win|Lose
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
    def _get_win_probability(matrix: np.ndarray, start_state: str = 'Lv-Lv') -> float:
        """Calculate probability of winning from a given state using absorbing Markov chain analysis."""
        state_index = GAME_STATES.index(start_state)
        
        # Split matrix into transient (Q) and absorbing (R) parts
        # Last 2 states are absorbing (Win, Lose)
        Q = matrix[:-2, :-2]
        R = matrix[:-2, -2:]
        
        # Fundamental matrix: N = (I - Q)^-1
        I = np.eye(Q.shape[0])
        N = np.linalg.inv(I - Q)
        
        # Absorption probabilities: B = N * R
        B = N @ R
        
        # Return probability of winning from start_state (first column is Win)
        return B[state_index, 0]
    
    def get_server_win_probability(self, start_state: str = 'Lv-Lv') -> float:
        """Get probability of winning game when serving."""
        return self._get_win_probability(self.matrix_server, start_state)
    
    def get_returner_win_probability(self, start_state: str = 'Lv-Lv') -> float:
        """Get probability of winning game when returning."""
        return self._get_win_probability(self.matrix_returner, start_state)
    
    def print_server_matrix(self):
        """Print server transition matrix with color coding."""
        print(f"\nTransition Matrix for {self.player.name} (Serving):")
        self._print_matrix(self.matrix_server)
    
    def print_returner_matrix(self):
        """Print returner transition matrix with color coding."""
        print(f"\nTransition Matrix for {self.player.name} (Returning):")
        self._print_matrix(self.matrix_returner)
    
    @staticmethod
    def _print_matrix(matrix: np.ndarray):
        """Print matrix with color coding for values."""
        def parity(index: int) -> tuple:
            p = index % 2 == 0
            bg_color = '\x1b[100m' if p else '\x1b[40m'
            text_color = '\x1b[30m' if p else '\x1b[37m'
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
        header = '\x1b[1;40mState\x1b[0;40m|' + '|'.join(state.rjust(5) for state in GAME_STATES)
        print(header)
        
        # Rows
        for i, row in enumerate(matrix):
            bg_color, text_color = parity(i)
            row_str = f"{bg_color}{text_color}{GAME_STATES[i].ljust(5)}|"
            row_str += '|'.join(colorize(val, i) for val in row)
            row_str += '\x1b[0m'
            print(row_str)


# Example usage
if __name__ == "__main__":
    sab = Player(
        name="Sabalenka",
        service_win_p=0.599,
        return_win_p=0.448
    )
    
    kyg = Player(
        name="Kygrios",
        service_win_p=0.69,
        return_win_p=0.16
    )
    
    # Sabalenka
    sabalenka_game = GameMatrix(sab)
    sabalenka_game.print_server_matrix()
    sabalenka_game.print_returner_matrix()
    print(f"\nSabalenka's probability of winning a game when serving: "
          f"{sabalenka_game.get_server_win_probability():.4f}")
    print(f"Sabalenka's probability of winning a game when returning: "
          f"{sabalenka_game.get_returner_win_probability():.4f}")
    
    # Kygrios
    kyrgios_game = GameMatrix(kyg)
    kyrgios_game.print_server_matrix()
    kyrgios_game.print_returner_matrix()
    print(f"\nKygrios's probability of winning a game when serving: "
          f"{kyrgios_game.get_server_win_probability():.4f}")
    print(f"Kygrios's probability of winning a game when returning: "
          f"{kyrgios_game.get_returner_win_probability():.4f}")
