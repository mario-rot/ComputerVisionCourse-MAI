from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from typing import List, Dict, Tuple
import random
import numpy as np

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
      self.use_gris_apec = use_grid_spec
      self.fig = None
      self.axs = None

  def __len__(self):
    return len(imgs)

  def show(self):
    if not self.use_gris_apec:
      self.fig, self.axs = plt.subplots(self.rows, self.cols, figsize=self.figsize)
      if self.rows <= 1 or self.cols <= 1:
        for idx, img in enumerate(self.imgs):
          self.axs[idx].imshow(img, cmap=self.cmap)
          if self.axis:
            self.axs[idx].axis(self.axis)
          if self.titles:
            self.axs[idx].set_title(self.titles[idx])
      else:
        im_ind = 0
        for row in range(self.rows):
          for column in range(self.cols):
            self.axs[row][column].imshow(self.imgs[im_ind], cmap=self.cmap)
            if self.axis:
              self.axs[row][column].axis(self.axis)
            if self.titles:
              self.axs[row][column].set_title(self.titles[im_ind])
            im_ind += 1
    else:
      self.fig = plt.figure(constrained_layout=True, figsize=self.figsize)
      if not self.order:
        self.order = [[i, [j, j + 1]] for i in range(self.rows) for j in range(self.cols)]
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
          im.set_title(self.titles[n])
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

# function to plot images from matches function in a grid with a custom layout
def matches_grid(imgs, rows = 1, cols = 1, titles=None, order = None, figsize = (10,10), axis=False, autoTitles=False, rotations=False):
  fig = plt.figure(constrained_layout=True, figsize = figsize)
  if not order:
    order = [[i, [j,j+1]] for i in range(rows) for j in range(cols)]
  gs = GridSpec(rows, cols, figure=fig)
  keypoints1_l = []
  keypoints2_l = []
  matches_l = []
  if imgs[0].shape[2] == 4:
      k = rgb2gray(rgba2rgb(imgs[0]))
  else:
      k = rgb2gray(imgs[0])
  for img in imgs[1]:
    if img.shape[2] == 4:
      img = rgb2gray(rgba2rgb(img))
    else:
      img = rgb2gray(img)
    kp1,kp2,matches = get_ORB(k,img)
    keypoints1_l.append(kp1)
    keypoints2_l.append(kp2)
    matches_l.append(matches)
  for n,(i,j) in enumerate(zip(imgs[1],order)):
    if i.shape[2] == 4:
      i = rgb2gray(rgba2rgb(i))
    else:
      i = rgb2gray(i)
    im = fig.add_subplot(gs[j[0],j[1][0]:j[1][1]])
    feature.plot_matches(im, k, i, keypoints1_l[n], keypoints1_l[n], matches_l[n])
    if axis:
      im.axis('off')
    if titles:
      im.set_title(titles[n])
    if autoTitles:
      if type(autoTitles) == type(str('')):
        im.set_title(autoTitles + str(matches_l[n].shape[0]))
      else:
        im.set_title('Matches: ' + str(matches_l[n].shape[0]) + ' -- Degrees: ' + str(autoTitles[n]))