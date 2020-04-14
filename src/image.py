import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
from src.palette import Palette

class Image:
  def __init__(self, path, n_colors=8, iters=100):
    self.__path = path
    self.__n_colors = n_colors
    self.content = img_as_float(io.imread(path))

  def generate_color_clusters(self):
    palette = Palette(n_colors=self.__n_colors)
    palette.fit(self.content)
    return palette

  def get_superpixels(self, n_segments=200, compactness=40, max_iter=10):
    """Segments image using k-means clustering in <Color,x,y> space
    """
    return slic(image=self.content, n_segments=n_segments, compactness=compactness, max_iter=max_iter)

  def get_prob_by_superpixel(self, superpixels, palette, prob_palette, T):
    """Get list of color probability by superpixel
    """
    image = self.content
    # Get list of unique labels matching each superpixel
    unique_superpixels = np.unique(superpixels)
    n_superpixels = len(unique_superpixels)
    
    # Initialize empty array for p(color|superpixel)
    cond_prob = np.zeros((n_superpixels, len(palette)))
    
    # Iterate over superpixels
    for i, superpixel in enumerate(unique_superpixels):
      # Compute superpixel mean color
      superpixel_idx = np.where(superpixels == superpixel)
      mean_color = np.mean(image[superpixel_idx], axis=0)
      
      # Compute energy term of conditional probability of each color
      cond_prob[i] = np.exp(- np.linalg.norm(palette.values - mean_color, axis=-1) / T)
    
    # Multiply by color likelihood and normalize
    cond_prob = prob_palette * cond_prob
    cond_prob = cond_prob / np.sum(cond_prob, axis=-1)[:, None]
    return cond_prob

  def pixelize(self):
    """Returns a pixelized image
    """
    # Init
    T = 1
    image = np.zeros_like(self.content)
    superpixels = self.get_superpixels()
    palette = self.generate_color_clusters()
    prob_palette = np.zeros(len(palette))
    # faudrait remplir palette avec une première couleur après le fit ? 
    
    while T > Image.__Tf:
      # associate superpixels to colors in palette
      cond_prob = self.get_prob_by_superpixel(image, superpixels, palette, prob_palette, 1)
      break
      # refine colors in palette

      # if palette converged
      #   reduce temp
      #   expand palette



    
