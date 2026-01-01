import numpy as np

SYMBOLS = np.array([1,2,3,4])
bloc1 = np.zeros((2,2), dtype=int)

b = np.array([[1,0],[0,0]])

for i in range(1,5):
    if i != b[0,0] and i != b[1,1] and i != b[1,0]: 
        b[0,1] = i 
        for j in range(1,5):
            if j != b[0,0] and j != b[1,1] and j != b[0,1]:
                b[1,0] = j
                for k in range(1,5):
                    if k != b[0,0] and k != b[1,0] and k != b[0,1]:
                        b[1,1] = k
                        print(b)
                        b[1,1] = 0
                b[1,0] = 0
        b[0,1] = 0
    
