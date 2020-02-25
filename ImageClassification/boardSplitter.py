import cv2
import numpy as np
import os


def detect_board_from_image(input_image: np.ndarray) -> np.ndarray:
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


def clean_and_make_square(image: np.ndarray) -> np.ndarray:
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


def resize_board(image: np.ndarray, size: tuple) -> np.ndarray:
    ''' Return a board that is the specified size'''
    if image.shape > size:
        resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
    return resized


def split_board(image: np.ndarray) -> dict:
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
            BOARD[postion] = a
            y += 1
        x += 1
    return BOARD


pwd = os.path.dirname(os.path.realpath(__file__))
img = cv2.imread(os.path.join(pwd, 'test_boards', '7.png'))

chessboard_extract = detect_board_from_image(img)
chessboard_clean = clean_and_make_square(chessboard_extract)
chessboard_resized = resize_board(chessboard_clean, (400, 400))
image_pos = split_board(chessboard_resized)
# for k,v in image_pos.items():
#     cv2.imwrite(os.path.join(pwd, 'squares', f'{k}.png'), v)