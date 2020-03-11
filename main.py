from sys import argv
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np

def load_image(path):
  print('Opening the image')
  return Image.open(path).convert('RGB')

def get_thumbnail(image, size=(32,32)):
  print('Making the thumbnail')
  thumbnail = image.copy()
  thumbnail.thumbnail(size)
  return thumbnail

def get_color_palette(image, k=8):
  print('Getting the color palette')
  data = list(image.getdata()) # flatten the shit out of it
  kmeans = KMeans(n_clusters = k)
  kmeans.fit(data)
  return [(int(color[0]), int(color[1]), int(color[2])) for color in kmeans.cluster_centers_]

def color_dist(colorA, colorB):
  delta_r = colorA[0] - colorB[0]
  delta_g = colorA[1] - colorB[1]
  delta_b = colorA[2] - colorB[2]
  return np.sqrt((delta_r ** 2) + (delta_g ** 2) + (delta_b ** 2))

def nearest_color(color, palette):
  # return color
  min = palette[0]
  min_dist = color_dist(color, palette[0])

  for palette_color in palette[1:]:
    dist = color_dist(color, palette_color)
    if dist < min_dist:
      min = palette_color
      min_dist = dist
  return min

def all_square_pixels(row, col, square_h, square_w):
  # Every pixel for a single "square" (superpixel)
  # Note that different squares might have different dimensions in order to
  # not have extra pixels at the edge not in a square. Hence: int(round())
  for y in range(int(round(row*square_h)), int(round((row+1)*square_h))):
    for x in range(int(round(col*square_w)), int(round((col+1)*square_w))):
      yield y, x

def make_one_square(image, row, col, square_h, square_w, palette):
    # Sets all the pixels in image for the square given by (row, col) to that
    # square's average color
    pixels = []
    # get all pixels
    for y, x in all_square_pixels(row, col, square_h, square_w):
      pixels.append(image[y][x])

    # get the average color
    av_r = 0
    av_g = 0
    av_b = 0
    for r, g, b in pixels:
      av_r += r
      av_g += g
      av_b += b
    av_r /= len(pixels)
    av_g /= len(pixels)
    av_b /= len(pixels)

    # set all pixels to the nearest color to that average color
    for y, x in all_square_pixels(row, col, square_h, square_w):
      image[y][x] = nearest_color((av_r, av_g, av_b), palette)

def recreate_image(image, num_cols, num_rows, palette):
  print('Recreating the image')
  data = np.array(image)
  (height, width, _) = data.shape
  square_w = width / num_cols
  square_h = height / num_rows
  
  for row in range(num_rows):
    for col in range(num_cols):
      make_one_square(data, row, col, square_h, square_w, palette)

  return Image.fromarray(data)

def save_image(image, path):
  print('Saving the recreated image')
  filepath_parts = path.rsplit('.', 1)
  filepath_parts[0] += '_pixelated'
  filepath = '.'.join(filepath_parts)
  image.save(filepath)

def main(path, num_rows, num_cols):
  image = load_image(path)
  thumbnail = get_thumbnail(image)
  palette = get_color_palette(thumbnail) # easier to calculate the color palette on thumbnail

  recreated_image = recreate_image(image, num_rows, num_cols, palette)
  save_image(recreated_image, path)


if (len(argv) == 4):
  main(argv[1], int(argv[2]), int(argv[3]))
else:
  print("Please provide good args bro")