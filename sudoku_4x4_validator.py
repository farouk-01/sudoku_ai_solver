import numpy as np
S_D = {1, 2, 3, 4}

def valid_cell(cell):
    return cell in S_D

def valid_row(row):
    return set(row) == S_D

def valid_col(col):
    return set(col) == S_D

def valid_block(block):
    return set(block) == S_D

    
