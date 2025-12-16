import numpy as np
import pickle

BOARD_ROWS = 3
BOARD_COLS = 3


class State:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.reset()

    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.playerSymbol = 1
        self.isEnd = False

    def getHash(self):
        return str(self.board.reshape(BOARD_ROWS * BOARD_COLS))

    def availablePositions(self):
        return list(zip(*np.where(self.board == 0)))

    def updateState(self, position):
        self.board[position] = self.playerSymbol
        self.playerSymbol *= -1

    def winner(self):
        for i in range(BOARD_ROWS):
            if abs(sum(self.board[i, :])) == 3:
                return np.sign(sum(self.board[i, :]))
            if abs(sum(self.board[:, i])) == 3:
                return np.sign(sum(self.board[:, i]))

        diag1 = sum(self.board[i, i] for i in range(BOARD_COLS))
        diag2 = sum(self.board[i, BOARD_COLS - i - 1] for i in range(BOARD_COLS))
        if abs(diag1) == 3:
            return np.sign(diag1)
        if abs(diag2) == 3:
            return np.sign(diag2)

        if not self.availablePositions():
            return 0
        return None

    def giveReward(self, result):
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(0.5)
            self.p2.feedReward(0.5)

    def train(self, rounds=50000):
        for _ in range(rounds):
            self.reset()
            while True:
                positions = self.availablePositions()
                p1_action = self.p1.chooseAction(positions, self.board, 1)
                self.updateState(p1_action)
                self.p1.addState(self.getHash())

                win = self.winner()
                if win is not None:
                    self.giveReward(win)
                    self.p1.reset()
                    self.p2.reset()
                    break

                positions = self.availablePositions()
                p2_action = self.p2.chooseAction(positions, self.board, -1)
                self.updateState(p2_action)
                self.p2.addState(self.getHash())

                win = self.winner()
                if win is not None:
                    self.giveReward(win)
                    self.p1.reset()
                    self.p2.reset()
                    break

        self.p1.savePolicy()
        print("Training Complete")

    def playHuman(self):
        self.reset()
        while True:
            positions = self.availablePositions()
            p1_action = self.p1.chooseAction(positions, self.board, 1)
            self.updateState(p1_action)
            self.showBoard()

            win = self.winner()
            if win is not None:
                print("Computer wins!" if win == 1 else "Tie!")
                break

            positions = self.availablePositions()
            p2_action = self.p2.chooseAction(positions)
            self.updateState(p2_action)
            self.showBoard()

            win = self.winner()
            if win is not None:
                print("Human wins!" if win == -1 else "Tie!")
                break

    def showBoard(self):
        print("-------------")
        for i in range(BOARD_ROWS):
            row = "| "
            for j in range(BOARD_COLS):
                if self.board[i, j] == 1:
                    row += "x | "
                elif self.board[i, j] == -1:
                    row += "o | "
                else:
                    row += "  | "
            print(row)
            print("-------------")


class Player:
    def __init__(self, name, exp_rate=0.3):
        self.name = name
        self.exp_rate = exp_rate
        self.lr = 0.2
        self.gamma = 0.9
        self.states = []
        self.states_value = {}

    def getHash(self, board):
        return str(board.reshape(BOARD_ROWS * BOARD_COLS))

    def chooseAction(self, positions, board, symbol):
        if np.random.rand() <= self.exp_rate:
            return positions[np.random.choice(len(positions))]

        best_value = -1e9
        action = None
        for p in positions:
            next_board = board.copy()
            next_board[p] = symbol
            value = self.states_value.get(self.getHash(next_board), 0)
            if value >= best_value:
                best_value = value
                action = p
        return action

    def addState(self, state):
        self.states.append(state)

    def feedReward(self, reward):
        for s in reversed(self.states):
            self.states_value[s] = self.states_value.get(s, 0) + \
                self.lr * (self.gamma * reward - self.states_value.get(s, 0))
            reward = self.states_value[s]

    def reset(self):
        self.states = []

    def savePolicy(self):
        with open("policy_" + self.name, "wb") as f:
            pickle.dump(self.states_value, f)

    def loadPolicy(self, file):
        with open(file, "rb") as f:
            self.states_value = pickle.load(f)


class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def chooseAction(self, positions):
        while True:
            row = int(input("Row (0-2): "))
            col = int(input("Col (0-2): "))
            if (row, col) in positions:
                return (row, col)


if __name__ == "__main__":
    # TRAIN
    p1 = Player("p1")
    p2 = Player("p2")
    state = State(p1, p2)
    state.train()

    # PLAY WITH HUMAN
    p1 = Player("p1", exp_rate=0)
    p1.loadPolicy("policy_p1")
    p2 = HumanPlayer("human")

    state = State(p1, p2)
    while input("Play again? (y/n): ").lower() == "y":
        state.playHuman()
