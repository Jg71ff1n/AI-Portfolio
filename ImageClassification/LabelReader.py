import numpy as np
import os

# path = 'c:\\projects\\hc2\\'

# files = []
# # r=root, d=directories, f = files
# for r, d, f in os.walk(path):
#     for file in f:
#         if '.jpeg' in file:
#             files.append(os.path.join(r, file))


# import glob

# path = 'c:\\projects\\hc2\\'

# files = [f for f in glob.glob(path + "**/*.txt", recursive=True)]

name = '1b1b1b2-3r4-1rK4b-R7-R2R1k2-2Bp4-2P5-2r5'

def readLabel(name:str):
    # 0,0 is top right corner 8,8 is bottom left
    board = np.zeros((8, 8), dtype=str)
    sections = name.split('-')
    row = 0
    while row < len(sections):
        section = sections[row]
        column = 0
        for i in section:
            number = 0
            try:
                number = int(i)
                for j in range(0, number):
                    board[row, column] = '0'
                    column += 1
            except ValueError:
                board[row, column] = i
                column += 1
        row += 1
    return board

print(readLabel(name))