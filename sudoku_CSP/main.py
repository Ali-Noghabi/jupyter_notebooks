import time
import math
import tkinter as tk  # draw gui
import example  # sudoku examples
N = 9 #change this variable base on you sudoku example
sqrtN = int(math.sqrt(N))
puzzle = example.puzzle2

# implement sudoku cell to store available numbers of every cell
class SudokuCell:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.value = 0
        self.available_numbers = set(range(1, N+1))

    def remove_number(self, num):
        if num in self.available_numbers:
            self.available_numbers.remove(num)

    def get_available_numbers(self):
        return self.available_numbers.copy()


class SudokuGrid:
    def __init__(self):
        self.grid = [[SudokuCell(row, col) for col in range(N)]
                     for row in range(N)]

        # initialize sub-grids
        self.subgrids = []
        for i in range(sqrtN):
            for j in range(sqrtN):
                subgrid = []
                for ii in range(sqrtN):
                    for jj in range(sqrtN):
                        cell = self.grid[i*sqrtN+ii][j*sqrtN+jj]
                        subgrid.append(cell)
                self.subgrids.append(subgrid)

    # to update available numbers of cells
    def set_value(self, row, col, value):
        cell = self.grid[row][col]
        cell.value = value

        # update available numbers of cells in same row
        for i in range(N):
            if i != col:
                cell = self.grid[row][i]
                cell.remove_number(value)

        # update available numbers of cells in same column
        for i in range(N):
            if i != row:
                cell = self.grid[i][col]
                cell.remove_number(value)

        # update available numbers of cells in same sub-grid
        subgrid = self.subgrids[(row//sqrtN)*sqrtN + (col//sqrtN)]
        for cell in subgrid:
            if cell.row != row and cell.col != col:
                cell.remove_number(value)

    # to restore available numbers in backtrack step
    def restore_value(self, row, col):
        cell = self.grid[row][col]
        value = cell.value
        cell.value = 0

        # restore available numbers of cells in same row
        for i in range(N):
            if i != col:
                cell = self.grid[row][i]
                cell.available_numbers.add(value)

        # restore available numbers of cells in same column
        for i in range(N):
            if i != row:
                cell = self.grid[i][col]
                cell.available_numbers.add(value)

        # restore available numbers of cells in same sub-grid
        subgrid = self.subgrids[(row//sqrtN)*sqrtN + (col//sqrtN)]
        for cell in subgrid:
            if cell.row != row and cell.col != col:
                cell.available_numbers.add(value)

    def is_valid_move(self, row, col, value):
        # check row
        for i in range(N):
            if i != col and self.grid[row][i].value == value:
                return False

        # check column
        for i in range(N):
            if i != row and self.grid[i][col].value == value:
                return False

        # check sub-grid
        subgrid = self.subgrids[(row//sqrtN)*sqrtN + (col//sqrtN)]
        for cell in subgrid:
            if cell.row != row and cell.col != col and cell.value == value:
                return False

        # forward checking
        # check row
        for i in range(N):
            if i != col and self.grid[row][i].value == 0:
                available_numbers = self.grid[row][i].get_available_numbers()
                if len(available_numbers) == 1 and value in available_numbers:
                    return False

        # check column
        for i in range(N):
            if i != row and self.grid[i][col].value == 0:
                available_numbers = self.grid[i][col].get_available_numbers()
                if len(available_numbers) == 1 and value in available_numbers:
                    return False

        # check sub-grid
        subgrid = self.subgrids[(row//sqrtN)*sqrtN + (col//sqrtN)]
        for cell in subgrid:
            if cell.row != row and cell.col != col and cell.value == 0:
                available_numbers = cell.get_available_numbers()
                if len(available_numbers) == 1 and value in available_numbers:
                    return False

        # move is valid
        return True


class SudokuSolver:
    def __init__(self, grid):
        self.grid = grid
        self.stack = []
        self.solution = None

    def solve(self):
        row, col = self.find_first_empty_cell()
        if row is None:
            # no more empty cells, puzzle is solved
            self.solution = self.grid
            return True

        cell = self.grid.grid[row][col]
        for value in cell.get_available_numbers():
            if self.grid.is_valid_move(row, col, value):
                self.stack.append((row, col, value))
                self.grid.set_value(row, col, value)
                if self.solve():
                    return True
                else:
                    self.grid.restore_value(row, col)
                    self.stack.pop()

        # no valid moves found, backtrack
        return False

    def find_first_empty_cell(self):
        for row in range(N):
            for col in range(N):
                if self.grid.grid[row][col].value == 0:
                    return row, col
        return None, None


class SudokuDrawer(tk.Canvas):
    def __init__(self, parent, board):
        self.WIDTH = 40
        self.HEIGHT = 40
        self.MARGIN = 5
        tk.Canvas.__init__(self, parent, width=self.WIDTH*N +
                           self.MARGIN*(N+1), height=self.HEIGHT*N+self.MARGIN*(N+1))
        self.parent = parent
        self.board = board
        self.draw_board()

    def draw_board(self):
        for row in range(N):
            for col in range(N):
                x0 = col*self.WIDTH + (col+1)*self.MARGIN
                y0 = row*self.HEIGHT + (row+1)*self.MARGIN
                x1 = x0 + self.WIDTH
                y1 = y0 + self.HEIGHT
                self.create_rectangle(
                    x0, y0, x1, y1, fill="white", tags="cell")
                if self.board[row][col].value != 0:
                    if puzzle[row][col] == 0:
                        self.create_text(
                            x0+self.WIDTH/2, y0+self.HEIGHT/2, text=str(self.board[row][col].value), font=("Arial", 20), tags="text")
                    else:
                        self.create_text(
                            x0+self.WIDTH/2, y0+self.HEIGHT/2, text=str(self.board[row][col].value), font=("Arial", 20), fill="blue", tags="text")


# initialize SudokuGrid object from puzzle
grid = SudokuGrid()
for row in range(N):
    for col in range(N):
        value = puzzle[row][col]
        if value != 0:
            grid.set_value(row, col, value)

start_time = time.time()
# solve puzzle using backtracking algorithm
solver = SudokuSolver(grid)
solver.solve()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"calculate time: {elapsed_time} secends")

# draw sudoku gui
root = tk.Tk()
root.title("Sudoku Board")
board = SudokuDrawer(root, solver.solution.grid)
board.pack()
root.mainloop()

#ChatGPT3.5 has been used in this code implement
