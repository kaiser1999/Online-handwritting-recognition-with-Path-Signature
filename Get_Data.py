import numpy as np
import struct
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

uint8_2_num = lambda x: sum([j << (i*8) for i, j in enumerate(x)])
uint8_2_str = lambda x: "".join(list(map(chr, x))).rstrip('\x00')

X_MARGIN, Y_MARGIN, HEIGHT, WIDTH = 20, 5, 126, 2400

def Equal_Distance_Sampling(X, t=0.5):
    '''
    Parameters
    ----------
    X : Stroke (x, y) in numpy Nx2 array
    t : fixed Euclidean distance between new adjacent points
    
    ----------
    Turn Stroke Uniform-Time to Equal-Distance Sampling
    1. Compute Euclidean distance between adjacent points
    2. Determine the number of points (num_pt) added between adjacent points via t
    3. Recompute the Euclidean distance between new points required for that stroke
    4. Add num_pt-1 (exclude original point) points based on distance in (3.)
    '''
    # 
    Euclidean = np.sqrt((np.diff(X, axis=0) ** 2).sum(axis=1))
    Z = []
    for i, num_pt in enumerate(Euclidean // t + 1):
        [Z.append(d / num_pt * (X[i+1] - X[i]) + X[i]) for d in np.arange(num_pt)]

    Z.append(X[-1])
    return np.array(Z)

#import esig.tosig as ts
import iisignature as iisig

def get_line_signature(x_coord, y_coord, window=4, t=0.5, sig_order=3):
    signature, X_eq_dist = [], []
    # loop each stroke in a line
    for x_, y_ in zip(x_coord, y_coord):
        # Uniform Time Sampling to Equal Distance Sampling
        Z = Equal_Distance_Sampling(np.vstack((x_, y_)).T, t=t)
        X_eq_dist.append(Z)
        sig = []
        for i in range(len(Z)):
            # Compute path signature for current point via sliding window approach
            t_1, t_2 = max(0, i-window), min(len(Z), i+window) + 1
            #sig.append(ts.stream2sig(Z[t_1:t_2], 3))
            sig.append(np.append(1, iisig.sig(Z[t_1:t_2], sig_order)))
        # sig: num of points x 15 (3rd order path signature)
        signature.append(np.array(sig))
    
    return signature, X_eq_dist

#%%
import cv2

def get_rotate_info(bitmap, scale=1, rotate_threshold=4):
    '''
    Parameters
    ----------
    bitmap : numpy HEIGHT x WIDTH array with values in {0, 255} for {black, white}
    scale : positive numeric for the scale of the box detected 

    Returns
    -------
    angle : angle of rotation in degree pi
    M : Affine transformation 2 x 3 matrix
    
    ----------
    Use a rectangle to frame all characters and compute the rotation angle 
    from the bottom right corner from 0 to 90
    
    '''
    coords = np.column_stack(np.where(bitmap > 0))
    *_, angle = cv2.minAreaRect(coords)
    if angle > 45:
        angle = 90 - angle
    else:
        angle = -angle
    
    if abs(angle) > rotate_threshold:
        angle = 0 # ensure it does not over rotate
        
    center = (WIDTH // 2, HEIGHT // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return angle, M

padding = lambda label, length, val: np.pad(label, (0, length - len(label)), constant_values=val)

#%%
from tensorflow.keras.backend import ctc_decode
import tensorflow as tf

class Preprocess():
    def __init__(self, path, extension=".wptt", max_length=None, sig_order=3):
        self.extension = extension
        self.raw_labels = []
        self.sig_order = sig_order
        
        files = [filepath for filepath in os.listdir(path) if extension in filepath]
        for filepath in tqdm(files):
            self.read_file(os.path.join(path, filepath))
            self.raw_labels = self.raw_labels + self.line_labels
        
        characters = set(char for label in self.raw_labels for char in label)
        self.characters = list(characters)

        self.max_length = max([len(label) for label in self.raw_labels])
        if max_length is not None:
            self.max_length = max(self.max_length, max_length)
        
        self.total_entries = len(self.raw_labels)
        self.input_shape = (HEIGHT, WIDTH, 2**(self.sig_order+1)-1)
    
    def read_file(self, filepath):
        '''
        See the pdf document http://www.nlpr.ia.ac.cn/databases/Download/WPTTRead.cpp.pdf
        '''
        self.line_labels = []
        with open(filepath, "rb") as f:
            head_uint8 = np.fromfile(f, dtype="uint8", count=4)
            header_size = uint8_2_num(head_uint8)
            
            format_uint8 = np.fromfile(f, dtype="uint8", count=8)
            illustration_uint8 = np.fromfile(f, dtype="uint8", count=header_size-54)
            
            type_uint8 = np.fromfile(f, dtype="uint8", count=20)
            length_uint8 = np.fromfile(f, dtype="uint8", count=2)
            code_length = uint8_2_num(length_uint8)
            data_uint8 = np.fromfile(f, dtype="uint8", count=20)  # short
            
            sample_uint8 = np.fromfile(f, dtype="uint8", count=4)
            page_uint8 = np.fromfile(f, dtype="uint8", count=4)
            
            stroke_uint8 = np.fromfile(f, dtype="uint8", count=4)
            stroke_num = uint8_2_num(stroke_uint8)
            
            stroke_x, stroke_y = [], []
            for i in range(stroke_num):
                point_uint8 = np.fromfile(f, dtype="uint8", count=2)
                point_num = uint8_2_num(point_uint8)
                
                x_coordinate, y_coordinate = [], []
                for j in range(point_num):
                    x_uint8 = np.fromfile(f, dtype="uint8", count=2)
                    y_uint8 = np.fromfile(f, dtype="uint8", count=2)
                    
                    x_coordinate.append(0.1*uint8_2_num(x_uint8))
                    y_coordinate.append(-0.1*uint8_2_num(y_uint8))
                    
                stroke_x.append(np.array(x_coordinate))
                stroke_y.append(np.array(y_coordinate))
        
            line_uint8 = np.fromfile(f, dtype="uint8", count=2)
            line_num = uint8_2_num(line_uint8)
            line_num_lst = []
            for i in range(line_num):
                line_stroke_uint8 = np.fromfile(f, dtype="uint8", count=2)
                line_stroke_num = uint8_2_num(line_stroke_uint8)
                line_idx = []
                for j in range(line_stroke_num):
                    line_idx_uint8 = np.fromfile(f, dtype="uint8", count=2)
                    line_idx.append(uint8_2_num(line_idx_uint8))
                line_num_lst.append(line_idx)
                
                line_char_uint8 = np.fromfile(f, dtype="uint8", count=2)
                line_char_num = uint8_2_num(line_char_uint8)
                
                line_label = ""
                for i in range(line_char_num):
                    tag_uint8 = np.fromfile(f, dtype="uint8", count=code_length)
                    tag_num = uint8_2_num(tag_uint8)
                    line_label += struct.pack("I", tag_num).decode("gbk", "ignore")[0]
                
                # line character
                self.line_labels.append(line_label)
        
        return stroke_x, stroke_y, line_num_lst
    
    def read_path(self, path, plot_fig=False):
        path = path.decode("utf-8")
        for filepath in os.listdir(path):
            if not filepath.endswith(self.extension):
                continue
            '''
            Parameters
            ----------
            path : a encoded string
            plot_fig : boolean, whether to plot graph for each line
            Note: This function is a generator, use for loop and next() to plot
            
            1. rescale the (x, y) coordinates of a line
            2. compute path signature up to order 3 for all strokes
            3. correct rotation problem based on path signature order 0 (bitmap)
            '''
            filepath = os.path.join(path, filepath)
            x_rescale_lst, y_rescale_lst = self.image_rescaling(*self.read_file(filepath))
            if plot_fig:
                n_line = len(x_rescale_lst)
                fig, axes = plt.subplots(n_line, 1, figsize=(30, 2.5*n_line))
            
            for line, (x_rescale, y_rescale) in enumerate(zip(x_rescale_lst, y_rescale_lst)):
                signature, X_eq_dist = get_line_signature(x_rescale, y_rescale, 4, 0.5, self.sig_order)
                
                data = np.zeros(self.input_shape)
                for sig, pt in zip(signature, X_eq_dist):
                    x_, y_ = pt.T.astype(np.uint)
                    data[HEIGHT - y_, x_, :] = sig
                
                for i in range(data.shape[-1]):
                    mini, maxi = np.min(data[:,:,i]), np.max(data[:,:,i])
                    data[:,:,i] = (data[:,:,i] - mini) * 255 / (maxi - mini)
                    if i == 0:
                        angle, M = get_rotate_info(data[:,:,0], scale=1)
                    
                    data[:,:,i] = cv2.warpAffine(data[:,:,i], M, (WIDTH, HEIGHT))
                
                if plot_fig:
                    axes[line].imshow(data[:,:,0], cmap="binary")
                    axes[line].axis("off")
                    
                # label encoding
                labels = padding([self.characters.index(char) for char in self.line_labels[line]], 
                                 self.max_length, len(self.characters))
                yield tf.constant(data/255, tf.float32), tf.constant(labels, tf.float32)
    
    def image_rescaling(self, stroke_x, stroke_y, line_num_lst):
        '''
        Reset the x-y axes and rescale in HEIGHT x WIDTH
        ------------------------------------------------
        Since the number of characters per-line is different, 
        the max. width among all lines is considered as the common width
        
        '''
        x_length = 0
        for stroke_lst in line_num_lst:
            x_coord = list(map(stroke_x.__getitem__, stroke_lst))
            x_min = min([min(x) for x in x_coord]) - X_MARGIN
            x_max = max([max(x) for x in x_coord]) + X_MARGIN
            if x_max - x_min > x_length:
                x_length = x_max - x_min
        
        x_rescale_lst, y_rescale_lst = [], []
        for stroke_lst in line_num_lst:
            x_coord = list(map(stroke_x.__getitem__, stroke_lst))
            x_min = min([min(x) for x in x_coord]) - X_MARGIN
            
            y_coord = list(map(stroke_y.__getitem__, stroke_lst))
            y_min = min([min(y) for y in y_coord]) - Y_MARGIN
            y_max = max([max(y) for y in y_coord]) + Y_MARGIN
            
            x_rescale = [((x_arr - x_min)/x_length * WIDTH) for x_arr in x_coord]
            y_rescale = [((y_arr - y_min)/(y_max - y_min) * HEIGHT) for y_arr in y_coord]
            
            x_rescale_lst.append(x_rescale)
            y_rescale_lst.append(y_rescale)
        
        return x_rescale_lst, y_rescale_lst
    
    def label_decoder(self, y_hat):
        '''
            y_hat : an N x (total words + 1) numpy array
        '''
        dictionary = np.array(self.characters + ["_"])
        return ["".join(label).replace('_', "") for label in dictionary[y_hat.astype(np.int32)]]
    
    def model_decoder(self, y_hat):
        '''
            y_hat : an N x Time x (total words + 1) numpy array
        '''
        dictionary = np.array(self.characters + ["_"])
        input_len = np.ones(y_hat.shape[0]) * y_hat.shape[1]
        results = ctc_decode(y_hat, input_length=input_len, greedy=True)[0][0]
        return ["".join(label).replace('_', "") for label in dictionary[results]]