import numpy as np

def _get_conv_outsize(input_size: int, kernel_size: int,\
                     stride: int, pad: int):
    return (input_size + 2 * pad - kernel_size) // stride + 1

def _im2col(image: np.ndarray, kernel_shape: tuple[int],\
           stride: int=1, pad: int=0):

    assert image.ndim == 4
    '''
    N: Number of image
    C: Number of channel
    H: Height, number of rows per image
    W: Width, number of colums per image
    '''
    # Key constants and variables
    # blocks_per_image: number of blocks which an image would generate with a given kernel
    N, C, H, W = image.shape 
    KH, KW = kernel_shape
    OH = _get_conv_outsize(H, KH, stride, pad)
    OW = _get_conv_outsize(W, KW, stride, pad)
    blocks_per_image = OH * OW
    block_size = C * KH * KW
    columned_image = np.zeros((N * blocks_per_image, block_size), dtype=image.dtype)
    
    # Padding
    if pad > 0:
        pad_width = ((0, 0), (0, 0), (pad, pad), (pad, pad))
        padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)
    else:
        padded_image = image
    
    for i in range(blocks_per_image):
        h_start = (i // OH) * stride
        h_end = h_start + KH
        w_start = (i % OH) * stride
        w_end = w_start + KW
        block = padded_image[:, :, h_start:h_end, w_start:w_end]
        flatten_block = block.reshape(N, -1)
        columned_image[i::blocks_per_image] = flatten_block
    
    return columned_image


def _col2im(col: np.ndarray, restore_shape: tuple[int],\
           kernel_shape: tuple[int],\
           stride: int=1, pad: int=0,\
           mode: str='backward'):
    
    N, C, H, W = restore_shape
    KH, KW = kernel_shape
    OH = _get_conv_outsize(H, KH, stride, pad)
    OW = _get_conv_outsize(W, KW, stride, pad)
    padded_image = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=col.dtype)

    blocks_per_image = col.shape[0] // N
    for i in range(blocks_per_image):
        N_blocks = col[i::blocks_per_image].reshape(N, C, KH, KW)
        h_start = (i // OH) * stride
        h_end = h_start + KH
        w_start = (i % OH) * stride
        w_end = w_start + KW

        if mode == 'normal':
            padded_image[:, :, h_start:h_end, w_start:w_end] = N_blocks
        elif mode == 'backward':
            padded_image[:, :, h_start:h_end, w_start:w_end] += N_blocks
    
    if pad > 0:
        image = padded_image[:, :, pad:-pad, pad:-pad]
    else:
        image = padded_image
    return image

if __name__ == '__main__':
    x = np.random.randint(1, 9, size=(1, 1, 5, 5))
    print(x)
    col_x = _im2col(x, (3, 3), 2)
    print(col_x)
    restored_x = _col2im(col_x, x.shape, (3, 3), 2)
    print(restored_x)