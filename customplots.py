from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from typing import List, Dict, Tuple
import random
from utils import get_ORB
from skimage import feature
from skimage.color import rgb2gray, rgba2rgb

class custom_grids():
  """"
  Function to print images in a personalized grid of images according to a layout provided
  or a simple arrangment auto calculated according to the number of columns and rows provided
  it is also possible to add layers of effects to some images as squares, lines, etc.
  """
  def __init__(self,
             imgs: List,
             rows: int = 1,
             cols: int = 1,
             titles: List = None,
             order: List = None,
             figsize: Tuple = (10,10),
             axis: str = None,
             cmap: str = None,
             title_size: int = 12,
             use_grid_spec: bool = True
             ):
      self.imgs = imgs
      self.rows = rows
      self.cols = cols
      self.titles = titles
      self.order = order
      self.figsize = figsize
      self.axis = axis
      self.cmap = cmap
      self.title_size = title_size
      self.use_gris_apec = use_grid_spec
      self.fig = None
      self.axs = None

      if not self.order:
        self.order = [[i, [j, j + 1]] for i in range(self.rows) for j in range(self.cols)]

  def __len__(self):
    return len(imgs)

  def show(self):
    if not self.use_gris_apec:
      self.fig, self.axs = plt.subplots(self.rows, self.cols, figsize=self.figsize)
      if self.rows <= 1 or self.cols <= 1:
        for idx, img_match in enumerate(match_imgs):
          self.axs.imshow(img, cmap=self.cmap)
          if self.axis:
            self.axs.axis(self.axis)
          if self.titles:
            self.axs.set_title(self.titles[idx], fontsize=self.title_size)
      elif self.rows <= 1 or self.cols <= 1:
        for idx, img in enumerate(self.imgs):
          self.axs[idx].imshow(img, cmap=self.cmap)
          if self.axis:
            self.axs[idx].axis(self.axis)
          if self.titles:
            self.axs[idx].set_title(self.titles[idx], fontsize= self.title_size)
      else:
        im_ind = 0
        for row in range(self.rows):
          for column in range(self.cols):
            self.axs[row][column].imshow(self.imgs[im_ind], cmap=self.cmap)
            if self.axis:
              self.axs[row][column].axis(self.axis)
            if self.titles:
              self.axs[row][column].set_title(self.titles[im_ind], fontsize= self.title_size)
            im_ind += 1
    else:
      self.fig = plt.figure(constrained_layout=True, figsize=self.figsize)
      gs = GridSpec(self.rows, self.cols, figure=self.fig)
      for n, (i, j) in enumerate(zip(self.imgs, self.order)):
        im = self.fig.add_subplot(gs[j[0], j[1][0]:j[1][1]])
        if self.cmap:
          im.imshow(i, cmap=self.cmap)
        else:
          im.imshow(i)
        if self.axis:
          im.axis('off')
        if self.titles:
          im.set_title(self.titles[n], fontsize= self.title_size)
  def overlay_image(self, img_idx, overlays, cmp_colors = None, alphas = None):
    if not cmp_colors:
      plt_clrs = plt.colormaps()
      cmp_colors = [random.choice(plt_clrs) for i in range(len(img_idx))]
    if not alphas:
      alphas = [0.5 for i in range(len(img_idx))]
    elif len(alphas) < len(img_idx):
      alphas = [alphas[i%len(alphas)] for i in range(len(img_idx))]
    for o_idx, i_idx in enumerate(img_idx):
      self.fig.axes[i_idx].imshow(overlays[o_idx], cmap=cmp_colors[o_idx], alpha=alphas[o_idx])

  def add_rects(self, img_idx, rects, rect_clrs = None, linewidth=1, facecolor=False):
    if not rect_clrs:
      rect_clrs = [random.choice(list(mcolors.CSS4_COLORS.keys())) for i in range(len(rects[0]))]
    if facecolor:
      face_clrs = rect_clrs.copy()
    else:
      face_clrs  = ['none' for i in range(len(rects[0]))]
    for i_idx, img in enumerate(rects):
      for r_idx,rect in enumerate(img):
        rect = patches.Rectangle(rect[0], rect[1], rect[2], linewidth=linewidth, edgecolor=rect_clrs[r_idx],
                               facecolor=face_clrs[r_idx])
        self.fig.axes[img_idx[i_idx]].add_patch(rect)

  def match_points(self, img_idx, matches_idxs, autoTitles = None):
    keypoints1_l = []
    keypoints2_l = []
    matches_l = []
    img = self.grayChecker(self.imgs[img_idx])
    match_imgs = [self.grayChecker(self.imgs[idx]) for idx in matches_idxs]
    for match in match_imgs:
      kp1, kp2, matches = get_ORB(img, match)
      keypoints1_l.append(kp1)
      keypoints2_l.append(kp2)
      matches_l.append(matches)

    self.fig = plt.figure(constrained_layout=True, figsize=self.figsize)
    gs = GridSpec(self.rows, self.cols, figure=self.fig)
    for n, (i, j) in enumerate(zip(match_imgs, self.order)):
      im = self.fig.add_subplot(gs[j[0], j[1][0]:j[1][1]])
      feature.plot_matches(im, img, i, keypoints1_l[n], keypoints2_l[n], matches_l[n])
      if self.axis:
        im.axis('off')
      if self.titles:
        im.set_title(slf.titles[n], fontsize= self.title_size)
      if autoTitles:
        if autoTitles == True:
          im.set_title('Matches: ' + str(matches_l[n].shape[0]), fontsize= self.title_size)
        else:
          im.set_title(autoTitles[n] + ' - Matches: ' + str(matches_l[n].shape[0]), fontsize= self.title_size)
  @staticmethod
  def grayChecker(color_img):
    if len(color_img.shape) == 2:
      gray_img = color_img
    elif color_img.shape[2] == 3:
      gray_img = rgb2gray(color_img)
    elif color_img.shape[2] == 4:
      gray_img = rgb2gray(rgba2rgb(color_img))

    return gray_img
