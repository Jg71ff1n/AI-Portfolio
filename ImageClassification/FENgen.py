import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import re
import string


class TileExtractor():

    def __init__(self, image):
        chessboard_extract = self.detect_board_from_image(image)
        chessboard_clean = self.clean_and_make_square(chessboard_extract)
        chessboard_resized = self.resize_board(chessboard_clean, (400, 400))
        self.image_pos = self.split_board(chessboard_resized)

    def detect_board_from_image(self, input_image: np.ndarray) -> np.ndarray:
        # Convert to grayscale for better contour detection
        gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        # Blur to remove fine detail and smooth image
        blur = cv2.medianBlur(gray, 5)
        # Sharpen to increase edge definition
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
        # Adaptive threshold to create contrasting areas based on neighbouring area
        thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        # Close morph transform to remove holes in objects
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Finds all areas where there is a contrast in colour
        contours = cv2.findContours(
            close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        areas = [cv2.contourArea(contour)
                 for contour in contours]  # Calculate contour areas
        # Choose largest, chess board is typically the largest
        biggest_area = np.amax(areas)
        # Create bounding rectangle
        x, y, w, h = cv2.boundingRect(contours[areas.index(biggest_area)])
        chess_board = input_image[y:y+h, x:x+w]  # Create image
        return chess_board

    def clean_and_make_square(self, image: np.ndarray) -> np.ndarray:
        '''
        Take the input board, if image is square, do nothing.
        Otherwise compute average of all edges of the image, if average equals the first pixel of the edge,
        removes the row/column. Runs until image is square or averages no longer equal the first pixel.
        '''
        while True:
            # Check left side edge (col 0)
            y, x, z = image.shape
            average_colour = np.mean(image[:, 0])
            if average_colour == np.mean(image[0][0]):
                image = image[:, 1:]
            else:
                break
        while True:
            # Check right side edge (col image.shape[0])
            y, x, z = image.shape
            average_colour = np.mean(image[:, x-1])
            if average_colour == np.mean(image[y-1][x-1]):
                image = image[:, :x-1]
            else:
                break
        while True:
            # Check top side edge (col image.shape[0])
            y, x, z = image.shape
            average_colour = np.mean(image[0, :])
            if average_colour == np.mean(image[0][0]):
                image = image[1:, :]
            else:
                break
        while True:
            # Check bottom side edge (col image.shape[0])
            y, x, z = image.shape
            average_colour = np.mean(image[y-1:, :])
            if average_colour == np.mean(image[y-1][x-1]):
                image = image[:y-1, :]
            else:
                break
        return image

    def resize_board(self, image: np.ndarray, size: tuple) -> np.ndarray:
        ''' Return a board that is the specified size'''
        if image.shape > size:
            resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
        return resized

    def split_board(self, image: np.ndarray) -> dict:
        TILE_HEIGHT = 50
        TILE_WIDTH = 50
        COLUMNS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        BOARD = {}
        img_width, img_height, img_depth = image.shape
        x = 0
        for i in range(0, img_height, TILE_HEIGHT):
            y = 0
            for j in range(0, img_width, TILE_WIDTH):
                box = ()
                a = image[i:i+TILE_HEIGHT, j:j+TILE_WIDTH]
                postion = f'{COLUMNS[x]}{8-y}'
                # create dictionary of {'position': image }
                BOARD[postion] = Image.fromarray(
                    cv2.cvtColor(a, cv2.COLOR_BGR2RGB))  # Convert from BGR to RGB colour space
                y += 1
            x += 1
        return BOARD


class ChessBoard():
    def __init__(self):
        self.board = np.zeros((8, 8), dtype=str)

    def load_from_fen(self, fen: str):
        self.board = np.zeros((8, 8), dtype=str)
        sections = re.split('-|/', fen)
        row = 0
        while row < len(sections):
            section = sections[row]
            column = 0
            for i in section:
                number = 0
                try:
                    number = int(i)
                    for j in range(0, number):
                        self.board[row, column] = '0'
                        column += 1
                except ValueError:
                    self.board[row, column] = i
                    column += 1
            row += 1

    def load_from_array(self, array: np.ndarray):
        if array.shape != (8, 8):
            raise ValueError('Supplied board is not correct shape')
        else:
            self.board = array

    def add_piece(self, pos, piece):
        if len(pos) != 2:
            return 'Position not in correct format, must be LetterNumber format, i.e. D5'
        column = pos[0]
        row = pos[1]
        array_column = string.ascii_letters.index(str.lower(column))
        array_row = 8-int(row)
        self.board[array_column][array_row] = piece

    def create_fen(self):
        fen_string = ''
        for row in self.board:
            empty_space = 0
            for column in row:
                try:
                    number = int(column)
                    empty_space += 1
                except ValueError:
                    if empty_space != 0:
                        fen_string += str(empty_space)
                    fen_string += column
                    empty_space = 0
            if empty_space != 0:
                fen_string += str(empty_space)
            fen_string += '/'
        return fen_string[:-1]  # remove trailing /

    def move_piece(self, start_pos, end_pos):
        if len(start_pos) != 2 or len(end_pos) != 2:
            return 'Positions not in correct format, must be LetterNumber format, i.e. D5'
        start_column = string.ascii_letters.index(str.lower(start_pos[0]))
        start_row = 8-int(start_pos[1])
        end_column = string.ascii_letters.index(str.lower(end_pos[0]))
        end_row = 8-int(end_pos[1])

        piece = self.board[start_row][start_column]
        if piece == '0':
            return f'There no piece at {start_pos}'
        else:
            self.board[start_row][start_column] = '0'
            self.board[end_row][end_column] = piece
            return f'{piece} moved to {end_pos}'

    def __str__(self):
        output = ''
        row_counter = 0
        for row in self.board:
            for column in row:
                output += column
                output += '  '
            output = output[:-1]
            output += ' | ' + str(8-row_counter)
            row_counter += 1
            output += '\n'
        output += '-' * 22
        output += '\na  b  c  d  e  f  g  h'
        return output


class ImageReader():

    def __init__(self):
        pwd = os.path.dirname(os.path.realpath(__file__))
        self.model = tf.keras.models.load_model(
            os.path.join(pwd, '13d_256b_15e.h5'))

    def read_board(self, image: np.ndarray) -> ChessBoard:
        # Split it into the 64 seperate tiles -> boardSplitter.py
        tiles = TileExtractor(image).image_pos
        # Run classifier on tiles
        labels = sorted(['0', 'b', 'B', 'k', 'K', 'n',
                         'N', 'p', 'P', 'q', 'Q', 'r', 'R'])
        board = ChessBoard()
        for pos, img in tiles.items():
            image_array = img_to_array(img)
            image_array = image_array.reshape(1, 50, 50, 3).astype('float')
            image_array = image_array/255
            prediction = labels[np.argmax(self.model.predict(image_array))]
            # Generate board
            board.add_piece(pos, prediction)
        return board
