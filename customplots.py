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
  
  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from pylab import *
import itertools
from sklearn.metrics import confusion_matrix


def groupedBarPlot(data, xticks, title,legend=True,axislabels = False,width=0.35,figsize=(25,10), barLabel=False, png = False, pdf = False, colors = None, fsizes = False, axisLim = False, xtick_rot=False, bLconfs = ['%.2f', 14]):
    """Width recomendado para 2 barras agrupadas es 0.35, para 3 y 4 es 0.2
       Para usar el barLabel, debe ser una lista de listas por cada tipo,
       aun que sea solo una barra por paso en el eje x deber ser una lista contenida dentro de otra
       Las opciones para fsizes son:
            'font' --> controla el tamaño de los textos por defecto
            'axes' --> tamaño de fuente del titulo y las etiquetas del eje x & y
            'xtick' --> tamaño de fuente de los puntos en el eje x
            'ytick' --> tamaño de fuente en los puntos del eje y
            'legend --> controla el tamaño de fuente de la leyenda
            'figure' --> controla el tamaño de fuente del titulo de la figura
       """
    if fsizes:
        for key,size in fsizes.items():
            if key == 'font':
                plt.rc(key, size=size)
            elif key == 'axes':
                plt.rc(key, titlesize=size)
                plt.rc(key, labelsize=size)
            elif key in ['xtick','ytick']:
                plt.rc(key, labelsize=size)
            elif key == 'legend':
                plt.rc(key, fontsize=size)
            elif key == 'figure':
                plt.rc(key, titlesize=size)
    else:
        plt.rc('font', size=15)

    x = np.arange(len(xticks))
    if colors:
        cl = colors
    else:
        cl = clrs

    if figsize:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots()

    rects = {}
    if len(data) == 1:
        ldata = list(data.values())
        keys = list(data.keys())
        rects[keys[0]] = ax.bar(x, ldata[0], width, label=keys[0], color = cl)
    elif len(data) == 2:
        ldata = list(data.values())
        keys = list(data.keys())
        rects[keys[0]] = ax.bar(x + width/2, ldata[0], width, label=keys[0], color = cl[2])
        rects[keys[1]] = ax.bar(x - width/2, ldata[1], width, label=keys[1], color = cl[3])
    elif len(data) == 3:
        ldata = list(data.values())
        keys = list(data.keys())
        rects[keys[0]] = ax.bar(x, ldata[0], width, label=keys[0])
        rects[keys[1]] = ax.bar([i+width for i in x], ldata[1], width, label=keys[1])
        rects[keys[2]] = ax.bar([i+2*width for i in x], ldata[2], width, label=keys[2])
    elif len(data) == 4:
        ldata = list(data.values())
        keys = list(data.keys())
        rects[keys[0]] = ax.bar(x + width/2, ldata[0], width, label=keys[0], color = cl[0])
        rects[keys[1]] = ax.bar(x - width/2, ldata[1], width, label=keys[1], color = cl[1])
        rects[keys[2]] = ax.bar(x + 1.5*width, ldata[2], width, label=keys[2], color = cl[2])
        rects[keys[3]] = ax.bar(x - 1.5*width, ldata[3], width, label=keys[3], color = cl[3])

    # ax.patch.set_facecolor('red')
    ax.patch.set_alpha(0.0)

    if axislabels:
        ax.set_xlabel(axislabels[0])
        ax.set_ylabel(axislabels[1])

    ax.set_title(title)
    if len(data) == 3:
        ax.set_xticks(x+width)
    else:
        ax.set_xticks(x)
    if xtick_rot:
        ax.set_xticklabels(xticks, rotation = xtick_rot)
    else:
        ax.set_xticklabels(xticks)

    if legend:
        ax.legend(prop={"size":30})

    if barLabel:
#         error = ['Hola' for i in range(9)]
#         ax.bar_label(list(rects.values())[0], padding=3, labels=[ e for e in error])
        try:
            for j,i in enumerate(rects.values()):
                ax.bar_label(i, padding=3, labels=[barLabel[0][:].format(ldata[j][r], barLabel[j+1][r]) for r in range(len(ldata[0]))])
        except:
            for j,i in enumerate(rects.values()):
                ax.bar_label(i, padding=3, labels=['{}\n{:.2f}%'.format(ldata[j][r], barLabel[j][r]) for r in range(len(ldata[0]))])
    else:
        for i in rects.values():
            ax.bar_label(i, padding=3, fmt = bLconfs[0], fontsize = bLconfs[1])

    fig.tight_layout()

    if axisLim:
        for key,values in axisLim.items():
            if key == 'xlim':
                plt.xlim(values[0], values[1])
            elif key == 'ylim':
                plt.ylim(values[0], values[1])

    if png:
        plt.savefig(png + '.png', transparent=True)
    if pdf:
        plt.savefig(pdf + '.pdf', transparent=True)

    plt.show()
