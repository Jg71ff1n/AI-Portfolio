import numpy as np
import os

path = 'c:\\projects\\hc2\\'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.jpeg' in file:
            files.append(os.path.join(r, file))


import glob

path = 'c:\\projects\\hc2\\'

files = [f for f in glob.glob(path + "**/*.txt", recursive=True)]

name = '1b1b2k1-K2B1q2-R3B2p-3b1NR1-5p2-3N4-8-5N2'

for picture in dataset:
    name = 
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