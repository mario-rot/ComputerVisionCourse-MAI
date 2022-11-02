from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches

# function to plot images using matplotlib subplots
def grid_images(imgs, rows=1, cols=1, titles=None, figsize = (10,10), axis=False, cmap=False):
  fig, axs = plt.subplots(rows, cols, figsize = figsize)
  if rows <= 1 or cols <= 1:
    for idx, img in enumerate(imgs):
      axs[idx].imshow(img, cmap=cmap)
      if axis:
        axs[idx].axis(axis)
      if titles:
        axs[idx].set_title(titles[idx])
  else:
    imgs = np.reshape(imgs, (rows, cols))
    for row in range(len(imgs)):
      for column in range(len(imgs[0])):
        axs[row][column].imshow(imgs[row][column], cmap=cmap)
        if axis:
          axs[row][column].axis(axis)
        if titles:
          axs[row][column].set_title(titles[row][column])

# function to plot images in a grid with a custom layout
def custom_grid(imgs, rows = 1, cols = 1, titles=None, order = None, figsize = (10,10), axis=False, cmap=False, addRect=False):
  fig = plt.figure(constrained_layout=True, figsize = figsize)
  if not order:
    order = [[i, [j,j+1]] for i in range(rows) for j in range(cols)]
  gs = GridSpec(rows, cols, figure=fig)
  for n,(i,j) in enumerate(zip(imgs,order)):
    im = fig.add_subplot(gs[j[0],j[1][0]:j[1][1]])
    if cmap:
      im.imshow(i, cmap=cmap)
    else:
      im.imshow(i)
    if axis:
      im.axis('off')
    if titles:
      im.set_title(titles[n])
    if addRect:
      rect_e = patches.Rectangle(addRect[n][0], addRect[n][3], addRect[n][2], linewidth=1, edgecolor='r', facecolor='none')
      rect_d = patches.Rectangle(addRect[n][1], addRect[n][3], addRect[n][2], linewidth=1, edgecolor='g', facecolor='none')
      im.add_patch(rect_e)
      im.add_patch(rect_d)

# function to plot images in a grid with a custom layout
def custom_grid(imgs, rows = 1, cols = 1, titles=None, order = None, figsize = (10,10), axis=False, cmap=False, overlay=False):
  fig = plt.figure(constrained_layout=True, figsize = figsize)
  if not order:
    order = [[i, [j,j+1]] for i in range(rows) for j in range(cols)]
  gs = GridSpec(rows, cols, figure=fig)
  k = imgs[0]
  for n,(i,j) in enumerate(zip(imgs[1],order)):
    im = fig.add_subplot(gs[j[0],j[1][0]:j[1][1]])
    if cmap:
      im.imshow(i, cmap=cmap)
    else:
      im.imshow(i)
    if overlay:
      im.imshow(k, cmap='gray')
      im.imshow(i, alpha=overlay[n], cmap='gray')
    if axis:
      im.axis('off')
    if titles:
      im.set_title(titles[n])

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