from random import sample
from pathlib import Path
import numpy as np
from PIL import Image
import os
from LabelReader import readLabel


def split(image, board, outputDir, prefix):
    TILE_HEIGHT = 50
    TILE_WIDTH = 50
    img_width, img_height = image.size
    x = 0
    for i in range(0, img_height, TILE_HEIGHT):
        y = 0
        for j in range(0, img_width, TILE_WIDTH):
            box = (j, i, j+TILE_WIDTH, i+TILE_HEIGHT)
            a = image.crop(box)
            try:
                int(board[x, y])
            except:
                piece = board[x, y]
                location = os.path.join(outputDir, piece)
                os.makedirs(location, exist_ok=True)
                output = os.path.join(
                    location, f'{prefix}_{x}_{y}_{piece}.jpeg')
                a.save(output)
            y += 1
        x += 1


def split_include_empty(image, board, outputDir, prefix):
    TILE_HEIGHT = 50
    TILE_WIDTH = 50
    img_width, img_height = image.size
    x = 0
    for i in range(0, img_height, TILE_HEIGHT):
        y = 0
        for j in range(0, img_width, TILE_WIDTH):
            box = (j, i, j+TILE_WIDTH, i+TILE_HEIGHT)
            a = image.crop(box)
            piece = board[x, y]
            location = os.path.join(outputDir, piece)
            os.makedirs(location, exist_ok=True)
            output = os.path.join(
                location, f'{prefix}_{x}_{y}_{piece}.jpeg')
            a.save(output)
            y += 1
        x += 1


def generate_data(image_path, outputDir, prefix):
    image_name = image_path.name
    image = Image.open(image_path)
    board = readLabel(image_name.split('.')[0])
    split_include_empty(image, board, outputDir, prefix)


rootDirTrain = Path(
    '/home/joe/GitDrive/AI-Portfolio/ImageClassification/ImageDataset/train')

rootDirTest = Path(
    '/home/joe/GitDrive/AI-Portfolio/ImageClassification/ImageDataset/test')

# filesTrain = [f for f in rootDirTrain.glob('**/*.jpeg') if f.is_file()]
# filesTest = [f for f in rootDirTest.glob('**/*.jpeg') if f.is_file()]


pwd = os.path.dirname(os.path.realpath(__file__))
outputDirTrain = Path(os.path.join(pwd, 'piecesDatasetAll', 'train', '0'))
outputDirTest = Path(os.path.join(pwd, 'piecesDatasetAll', 'test', '0'))
filesTrain = [f for f in outputDirTrain.glob('**/*.jpeg') if f.is_file()]
filesTest = [f for f in outputDirTest.glob('**/*.jpeg') if f.is_file()]


# for i in range(0, len(filesTrain)):
#     current_file = filesTrain[i]
#     generate_data(current_file, outputDirTrain, i)
#     print(f'Working on training file: {current_file.name}, number: {i}')

# for i in range(0, len(filesTest)):
#     current_file = filesTest[i]
#     generate_data(current_file, outputDirTest, i)
#     print(f'Working on testing file: {current_file.name}, number: {i}')


train_to_delete = sample(filesTest, int(len(filesTest)*(1-0.02)))
for i in range(0, len(train_to_delete)):
    current_file = train_to_delete[i]
    os.remove(current_file)
