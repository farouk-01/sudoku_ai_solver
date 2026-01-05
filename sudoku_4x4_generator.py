import numpy as np


def is_in_block(grid, r, c, num):
    br = 2 * (r // 2)
    bc = 2 * (c // 2)
    return num in grid[br:br + 2, bc:bc +2]

def is_valid(grid, r, c, num):
    if num in grid[r] or num in grid[:,c]: 
        return False
    if is_in_block(grid, r, c, num): 
         return False
    return True
    
def all_solutions():
    grid = np.zeros((4,4), dtype=int)

    def generate_sudoku(cell=0):
        if cell == 16:
            yield grid.copy()
            return
        
        r, c = divmod(cell, 4)

        for v in (1, 2, 3, 4):
            if is_valid(grid, r, c, v):
                grid[r][c] = v
                yield from generate_sudoku(cell + 1)
                grid[r][c] = 0

    yield from generate_sudoku()

count = 0
for sol in all_solutions():
    count += 1
    print(f"Solution #{count}:")
    for row in sol:
                print(row)
    print("-----------")
print(f"Nombre de sudoku 4x4 trouvé : {count}")
