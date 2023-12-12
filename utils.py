import os
import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn


# def make_folder(path, version):
#     if not osp.exists(osp.join(path, version)):
#         os.makedirs(osp.join(path, version))


# def denorm(x):
#     out = (x + 1) / 2
#     return out.clamp_(0, 1)


# def uint82bin(n, count=8):
#     """returns the binary of integer n, count refers to amount of bits"""
#     return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])


# def labelcolormap(N):
#     if N == 19:  # CelebAMask-HQ
#         cmap = np.array([(0,  0,  0), (204, 0,  0), (76, 153, 0),
#                          (204, 204, 0), (51, 51, 255), (204, 0, 204), (0, 255, 255),
#                          (51, 255, 255), (102, 51, 0), (255, 0, 0), (102, 204, 0),
#                          (255, 255, 0), (0, 0, 153), (0, 0, 204), (255, 51, 153),
#                          (0, 204, 204), (0, 51, 0), (255, 153, 51), (0, 204, 0)],
#                         dtype=np.uint8)
#     else:
#         cmap = np.zeros((N, 3), dtype=np.uint8)
#         for i in range(N):
#             r, g, b = 0, 0, 0
#             id = i
#             for j in range(7):
#                 str_id = uint82bin(id)
#                 r = r ^ (np.uint8(str_id[-1]) << (7-j))
#                 g = g ^ (np.uint8(str_id[-2]) << (7-j))
#                 b = b ^ (np.uint8(str_id[-3]) << (7-j))
#                 id = id >> 3
#             cmap[i, 0] = r
#             cmap[i, 1] = g
#             cmap[i, 2] = b
#     return cmap


# class Colorize(object):
#     def __init__(self, n=19):
#         self.cmap = labelcolormap(n)
#         self.cmap = torch.from_numpy(self.cmap[:n])

#     def __call__(self, gray_image):
#         size = gray_image.size()
#         color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

#         for label in range(0, len(self.cmap)):
#             mask = (label == gray_image[0]).cpu()
#             color_image[0][mask] = self.cmap[label][0]
#             color_image[1][mask] = self.cmap[label][1]
#             color_image[2][mask] = self.cmap[label][2]

#         return color_image


# def tensor2label(label_tensor, n_label, imtype=np.uint8):
#     label_tensor = label_tensor.cpu().float()
#     label_tensor = Colorize(n_label)(label_tensor)
#     label_numpy = label_tensor.numpy()
#     label_numpy = label_numpy / 255.0
#     return label_numpy


# def generate_label(inputs, imsize, class_num=19):
#     '''Tensor after optimized...'''

#     inputs = F.interpolate(input=inputs, size=(imsize, imsize),
#                            mode='bilinear', align_corners=True)
#     pred_batch = torch.argmax(inputs, dim=1)
#     label_batch = torch.Tensor(
#         [tensor2label(p.view(1, imsize, imsize), class_num) for p in pred_batch])
#     return label_batch

# def generate_label_plain(inputs, imsize, class_num=19):
#     inputs = F.interpolate(input=inputs, size=(imsize, imsize),
#                            mode='bilinear', align_corners=True)
#     pred_batch = torch.argmax(inputs, dim=1)
#     label_batch = [p.cpu().numpy() for p in pred_batch]
#     return label_batch

# def generate_compare_results(images, labels, preds, imsize, class_num=19):
#     '''Tensor after optimized...'''
#     labels = F.interpolate(input=labels, size=(imsize, imsize),
#                            mode='bilinear', align_corners=True)
#     label_batch = torch.argmax(labels, dim=1)
#     labels_batch = torch.Tensor(
#         [tensor2label(p.view(1, imsize, imsize), class_num) for p in label_batch])
#     preds = F.interpolate(input=preds, size=(imsize, imsize),
#                            mode='bilinear', align_corners=True)
#     pred_batch = torch.argmax(preds, dim=1)
#     preds_batch = torch.Tensor(
#         [tensor2label(p.view(1, imsize, imsize), class_num) for p in pred_batch])
#     compare_batch = torch.cat((denorm(images).cpu().data, labels_batch, preds_batch), 3)
#     return compare_batch


# def adjust_learning_rate(g_lr, optimizer, i_iter, total_iters):
#     """The learning rate decays exponentially"""

#     def lr_poly(base_lr, iter, max_iter, power):
#         return base_lr * ((1 - float(iter) / max_iter) ** (power))

#     lr = lr_poly(g_lr, i_iter, total_iters, .9)
#     optimizer.param_groups[0]['lr'] = lr

#     return lr


# class AverageMeter(object):
#     """Computes and stores the average and current value"""

#     def __init__(self):
#         self.initialized = False
#         self.val = None
#         self.avg = None
#         self.sum = None
#         self.count = None

#     def initialize(self, val, weight):
#         self.val = val
#         self.avg = val
#         self.sum = val * weight
#         self.count = weight
#         self.initialized = True

#     def update(self, val, weight=1):
#         if not self.initialized:
#             self.initialize(val, weight)
#         else:
#             self.add(val, weight)

#     def add(self, val, weight):
#         self.val = val
#         self.sum += val * weight
#         self.count += weight
#         self.avg = self.sum / self.count

#     def value(self):
#         return self.val

#     def average(self):
#         return self.avg

# Perform one hot encoding on label
# def one_hot_encode(label, class_num):
#     """
#     Convert a segmentation image label array to one-hot format
#     by replacing each pixel value with a vector of length num_classes
#     # Arguments
#         label: The 2D array segmentation image label
#         label_values
        
#     # Returns
#         A 2D array with the same width and hieght as the input, but
#         with a depth size of num_classes
#     """
#     semantic_map = []
#     for cls in range(class_num):
#         equality = np.equal(label, cls)
#         class_map = np.all(equality, axis = -1)
#         semantic_map.append(class_map)
#     semantic_map = np.stack(semantic_map, axis=-1)

#     return semantic_map

def one_hot_encode(segmentation_map, num_classes=19):
    # Create an empty array with dimensions (num_classes, height, width)
    one_hot_mask = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], num_classes), dtype=np.uint8)
    
    # Iterate through each class and set the corresponding channel to 1
    for class_label in range(num_classes):
        one_hot_mask[:, :, class_label] = (segmentation_map == class_label).astype(np.uint8)
    
    return one_hot_mask
    
# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    x = np.argmax(image, axis = 0)
    return x