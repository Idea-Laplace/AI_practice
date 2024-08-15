import numpy as np
from dataclasses import dataclass, field
import os, sys
sys.path.append(os.pardir)
from mnist_data_preprocessing import l_image, l_index, batch_generator
from basic_layers import numerical_gradient


RESIDUE = -1

class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        self.mask = None
        self.input = None
    
    def forward(self, x):
        n, c, h, w = x.shape
        self.input = x
        out_h = (h + 2 * self.pad - self.pool_h) // self.stride + 1
        out_w = (w + 2 * self.pad - self.pool_w) // self.stride + 1
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        '''
        col - 2D schema
                ch1  ch2  ch3
                ---- ---- ----
                ---- ---- ----
        img1    ---- ---- ----
                ---- ---- ----
                ch1  ch2  ch3
                ---- ---- ----
                ---- ---- ----
        img2    ---- ---- ----
                ---- ---- ----
                ch1  ch2  ch3
                ---- ---- ----
                ---- ---- ----
        img3    ---- ---- ----
                ---- ---- ----
        '''
        col = col.reshape(-1, self.pool_h * self.pool_w)
        '''
        img 1   ch1-1 -*--
                ch2-1 ---*
                ch3-1 *---

                ch1-2 -*--
                ch2-2 -*--
                ch3-2 -*--

                ch1-3 ---*
                ch2-3 ---*
                ch3-3 -*--

                ch1-4 *---
                ch2-4 *---
                ch3-4 ---*
        '''

        self.mask = col != col.max(axis=1, keepdims=True)
        self.mask = self.mask.reshape(-1, c * self.pool_h * self.pool_w)
        self.mask = col2im(self.mask, self.input.shape, self.pool_h, self.pool_w,\
                           self.stride, self.pad)
        out = np.max(col, axis=1)
        '''
        img 1   ch1-1 *
                ch2-1 *
                ch3-1 *
                ch1-2 *
                ch2-2 *
                ch3-2 *
                ch1-3 *
                ch2-3 *
                ch3-3 *
                ch1-4 *
                ch2-4 *
                ch3-4 *
        '''
        out = out.reshape(n, out_h, out_w, c).transpose(0, 3, 1, 2)
        return out
    
    def backward(self, dout):
        dn, dc, dh, dw = dout.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, 1)
        dx = np.zeros((dout.size, self.pool_h * self.pool_w))
        dx += dout
        '''
        img 1   ch1-1 ****
                ch2-1 ****
                ch3-1 ****
                ch1-2 ****
                ch2-2 ****
                ch3-2 ****
                ch1-3 ****
                ch2-3 ****
                ch3-3 ****
                ch1-4 ****
                ch2-4 ****
                ch3-4 ****
        '''
        dx = dx.reshape(-1, dc * self.pool_h * self.pool_w)
        dx = col2im(dx, self.input.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        dx[self.mask.astype(bool)] = 0

        return dx


class Convolution:
    def __init__(self, input_size: tuple, filter_size: tuple, stride: int=1, pad: int=0):
        fan_in = np.prod(filter_size[1:])
        stddev = np.sqrt(2 / fan_in)
        self.w = np.random.normal(0, stddev, filter_size)
        self.b = np.zeros((filter_size[0], 1, 1))
        self.x = None
        self.col = None
        self.dw = None
        self.db = None
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        # f stands for filter, n stands for number,
        # w stands for width, h stands for height
        # c stands for channels
        # self.w itself is a filter
        fn, c, fh, fw = self.w.shape
        n, c, h, w = x.shape
        out_h = (h + 2 * self.pad - fh) // self.stride + 1
        out_w = (w + 2 * self.pad - fw) // self.stride + 1

        # What is complex is just making the input image batch into a properly-rearranged 2d array.
        # , which cosists of flattened 'blocks' that would be multiplicated by filters.
        self.x = x
        self.col = im2col(self.x, fh, fw, self.stride, self.pad)
        # Reshaping the set of filters is of no difficult.
        # Transpose operator is needed to implement element-wise multiplication without
        # changing the data structure of the filters and image sets.
        col_w = self.w.reshape(fn, RESIDUE).T

        # self.b addition is a kind of broad casting.
        # col: (row: number of images * blocks per image) x (col: filter size including channels)
        # col_w: (row: filter size including channels) x (col: number of channels)
        # np.dot(col, col_w): each row, consists of block * filter result, of the number of filter number.
        out = np.dot(self.col, col_w)
        '''
        b: block, f: filter
        [b1f1, b1f2, ..., b1fn]
        [b2f1, b2f2, ..., b2fn]
        .
        .
        .
        [bmf1, bmf2, ..., bmfn]
        '''
        # Each row in the 'out' should be of shape 'block', (fn, out_h, out_w)
        # -1 would be the number of filter number
        # From (N, H, W, C) to (N, C, H, W), this changes the output into another 'input'
        # for other Convolution classes.
        out = out.reshape(n, out_h, out_w, -1).transpose(0, 3, 1, 2)
        out += self.b
        
        return out
    
    def backward(self, dout):
        dn, dc, dh, dw = dout.shape
        fn, fc, fh, fw = self.w.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(dn * dh * dw, dc)

        self.dw = np.dot(self.col.T, dout)
        self.dw = self.dw.T
        self.dw = self.dw.reshape(fn, fc, fh, fw)

        self.db = np.sum(dout, axis=0)
        self.db = self.db.reshape(-1, 1, 1)

        # The forward method uses self.w.reshape(fn, RESIDUE).T,
        # hence, in the backward method, just the self.w.reshape(fn, RESIDUE) is used
        # which is same with self.w.reshape(fn, RESIDUE).T.T
        dcol = np.dot(dout, self.w.reshape(fn, RESIDUE))
        dx = col2im(dcol, self.x.shape, fh, fw, self.stride, self.pad)
        return dx


# image to column
def im2col(input: np.ndarray,\
           filter_columns: int,\
           filter_rows: int,\
           stride: int=1,\
           pad: int=0) -> np.ndarray:

    if input.ndim == 2:
        input = input.reshape(1, 1, *input.shape)
    elif input.ndim == 3:
        input = input.reshape(1, *input.shape)

    assert input.ndim == 4, 'The input data must be of shape of 4D'

    # Padding
    pad_width = ((0, 0), (0, 0), (pad, pad), (pad, pad))
    padded_batch = np.pad(input, pad_width, mode='constant', constant_values=0)

    # Determining the shape of the return array
    num, channel, height, width = padded_batch.shape
    out_height = (height - filter_rows) // stride + 1
    out_width = (width - filter_columns) // stride + 1
    out_area = out_height * out_width

    # np.zeros((Number of blocks, (block_dimension)))
    column = np.zeros((num * out_width * out_height, channel, filter_rows, filter_columns))
    block_row = 0
    for i in range(0, out_height):
        for j in range(0, out_width):
            x, y = i * stride, j * stride
            block = padded_batch[:, :, x:x+filter_rows, y:y+filter_columns]
            column[block_row::out_area] = block
            block_row += 1
    
    column = column.reshape(num * out_width * out_height, -1)

    return column


'''
dcol - 2D schema
         ch1  ch2  ch3
        ---- ---- ----
        ---- ---- ----
img1    ---- ---- ----
        ---- ---- ----
         ch1  ch2  ch3
        ---- ---- ----
        ---- ---- ----
img2    ---- ---- ----
        ---- ---- ----
         ch1  ch2  ch3
        ---- ---- ----
        ---- ---- ----
img3    ---- ---- ----
        ---- ---- ----

restore to: (images, channels, row, column)

'''
# column to image
# This version of the col2im is imperfect when the 'block' would not cover the whole image
# since the image could not be divided by blocks with the given stride.
# This problem should be resolved later.
def col2im(col: np.ndarray,\
            restore_shape: tuple,\
            filter_rows: int,\
            filter_columns: int,\
            stride: int=1,\
            pad: int=0):

    # 4D version
    n, c, h, w = restore_shape
    block_rows = (h + 2 * pad - filter_rows) // stride + 1
    block_cols = (w + 2 * pad - filter_columns) // stride + 1
    blocks_per_img = block_rows * block_cols

    img = np.ones((n, c, h + 2 * pad, w + 2 * pad))

    for i, row in enumerate(col):
        block = row.reshape(c, filter_rows, filter_columns)
        x = (stride * (i // block_cols)) % block_rows
        y = stride * (i % block_cols)
        img_row = i // blocks_per_img

        if block.dtype == bool:
            img[img_row, :, x:x+filter_rows, y:y+filter_columns] = \
                img[img_row, :, x:x+filter_rows, y:y+filter_columns].astype('bool') & block
            #print(block)
        else:
            img[img_row, :, x:x+filter_rows, y:y+filter_columns] = block
    
    if pad < 0:
        raise ValueError('pad value should be nonnegative.')
    elif pad > 0:
        img_padding_removed = img[:, :, pad:-pad, pad:-pad]
    else:
        img_padding_removed = img
    
    return img_padding_removed


        


if __name__ == '__main__':
    t = np.random.randint(0, 9, size=(8, 8, 8, 8))
    #print(t)
    col_t = im2col(t, 2, 2, 1, 0)
    #print(col_t)

    restored_t = col2im(col_t, (8, 8, 8, 8), 2, 2, 1, 0)
    #print(restored_t)

    test_pooling = Pooling(2, 2, 2)
    pooled = test_pooling.forward(t)
    print(pooled)
    print(t.shape)
    print(pooled.shape)
    back = test_pooling.backward(0.1 * pooled)
    print(back)