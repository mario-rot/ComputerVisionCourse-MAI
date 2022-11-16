from skimage import feature
from skimage.color import rgb2gray
import numpy as np

def get_ORB(img1, img2):
  descriptor_extractor = feature.ORB(n_keypoints=200)
  descriptor_extractor.detect_and_extract(img1)
  keypoints1 = descriptor_extractor.keypoints
  descriptors1 = descriptor_extractor.descriptors

  descriptor_extractor.detect_and_extract(img2)
  keypoints2 = descriptor_extractor.keypoints
  descriptors2 = descriptor_extractor.descriptors

  matches12 = feature.match_descriptors(descriptors1, descriptors2, cross_check=True)

  return keypoints1, keypoints2, matches12

def get_multi_ORB(de, imgs):
  keypoints = []
  descriptors = []
  for img in imgs:
    if len(img.shape) != 2:
      img = rgb2gray(img)
    de.detect_and_extract(img)
    keypoints.append(de.keypoints)
    descriptors.append(de.descriptors)
  return np.array(keypoints), np.array(descriptors)
