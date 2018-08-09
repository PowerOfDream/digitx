import numpy as np
from PIL import Image, ImageFont, ImageDraw
import glob
import os

def gen_background(size, mu, sigma):
    '''
    generate a background grayscale image
    size:  image size
    mu:    bright mean
    sigma: bright variance

    return Image
    '''

    pixel_array = sigma * np.random.randn(size[1], size[0]) + mu
    pixel_array[pixel_array > 255] = 255
    pixel_array[pixel_array < 0  ] = 0
    pixel_array = pixel_array.astype(np.uint8)

    return Image.fromarray(pixel_array, 'L')


def rend_char_img(char, angle, img_size, font_size, font_type):
    '''
    rend a character image
    char:      character to be rend
    angle:     character rotate angle
    img_size:  size of output image
    font_size: font size
    font_type: font type

    return character image, white background and black forecolor
    '''

    img = Image.new('L', img_size, 0)
    font = ImageFont.truetype(font_type, size = font_size)
    char_size = font.getsize(char)

    drawer = ImageDraw.Draw(img)
    drawer.text(((img_size[0] - char_size[0]) / 2, (img_size[1] - char_size[1]) / 2), char, font = font, fill = 255)
    del drawer
    
    img = img.rotate(angle)

    return img


def clip_box(box, edge, img_size):
    '''
    make a square clip box base on real box + edge
    box: real box
    img_size: clip box must less than img_size
    '''

    left, top, right, bottom = box
    left   -= edge[0]
    top    -= edge[1]
    right  += edge[2]
    bottom += edge[3]
    w = right - left
    h = bottom - top
    if (w > h):
        top    -= (w - h) / 2
        bottom += (w - h) / 2
    elif (w < h):
        left  -= (h - w) / 2
        right += (h - w) / 2
    if left < 0:
        left = 0
    if top < 0:
        top = 0
    if right > img_size[0]:
        right = img_size[0]
    if bottom > img_size[1]:
        bottom = img_size[1]

    out_box = (left, top, right, bottom)

    #print("box = ",  str(box))
    #print("edge = ", str(edge))
    #print("img_size = ", str(img_size))

    return out_box


def randomize_and_crop_char_img(char_img, mu, sigma, edge):
    '''
    add random to character image
    mu:    mena of character stroke is 255 * (1.0 - mu)
    sigma: variance of character stroke is 255 * sigma
    edge:  a tuple, 4 side edge after crop
    '''

    randmizer = sigma * np.random.randn(char_img.size[1], char_img.size[0]) + mu
    pix = np.array(char_img)
    
    #disable halftone, otherwise a white edge may sourround the character
    pix[pix > 0] = 255 
    
    pix = pix * randmizer
    pix[pix > 255] = 255
    pix[pix < 0  ] = 0

    pix = pix.astype(int)   
    pix = 255 - pix    #swith forecolor and background color
    pix = pix.astype(np.uint8)

    box = clip_box(char_img.getbbox(), edge, char_img.size)
    croped = Image.fromarray(pix, 'L').crop(box)
    mask = char_img.crop(box)
    
    return croped, mask


def merge_char_and_background(back_img, char_img, mask):
    '''
    merge char image onto back image

    back_img must char_img mask must be the same size
    '''
    
    back_arr = np.array(back_img)
    char_arr = np.array(char_img)
    mask_arr = np.array(mask)

    #print(back_arr.shape, char_arr.shape, mask_arr.shape)

    bit_mask = (mask_arr > 0).astype(int)
    pix = char_arr * bit_mask + back_arr * (1 - bit_mask)
    pix[pix > 255] = 255
    pix[pix < 0  ] = 0
    pix = pix.astype(np.uint8)

    merged_img = Image.fromarray(pix, 'L')
    
    return merged_img


def gen_char_img(char, angle, back_mean, back_var, fore_mean, fore_var, font_path, out_size = 32, margin = (0, 0, 0, 0)):
    '''
    char:    the character to be rended
    angle:   angle of the character rotation
    
    back_mean: white background bright mean, back_mean - back_var should be larger than fore_mean + fore_var
    back_var:  white bakcground bright variance
    
    fore_mean: black forecolor bright mean, back_mean - back_var should be larger than fore_mean + fore_var
    fore_var:  black forecolor bright variance 

    out_size: output image size, a integer, the real output image size is (out_size, out_size)

    margin:  4 side (left, top, right ,bottom) spaces
    '''
  
    origi_char_img = rend_char_img(char, angle, (out_size*4, out_size*4), out_size*3, font_path)
       
    char_img_with_rand, mask = randomize_and_crop_char_img(origi_char_img, (255.0 - fore_mean) / 255.0,  fore_var / 255.0, margin)
    
    background_img = gen_background(char_img_with_rand.size, back_mean, back_var)

    merged_img = merge_char_and_background(background_img, char_img_with_rand, mask)

    out_img = merged_img.resize((out_size, out_size), Image.BILINEAR)
    
    return out_img


def gen_batch_examples(batch_size, img_size, font_dir, output_path = None):
    '''
    Function: genrate a batch of examples
    Input:
        batch_size:   int, batch size of examples
        img_size:     int, image size of each examples
        font_dir:     str, directory for looking true type fonts

    output:
        X:  batch examples with shape(batch_size, img_size, img_size)
        Y:  label with shape(batch_size, 1)
    '''
    char_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'X']
    max_angle = 10
    min_back_mean = 60
    max_back_var_ratio = 0.3
    max_fore_mean = 200
    max_fore_var_ratio = 0.3
    font_list = glob.glob(font_dir + "/*.ttf")

    margin_list = [0,   1,   2,   3,    4,    5,    -1,  -2,    -3]
    probability = [0.2, 0.2, 0.1, 0.08, 0.09, 0.07, 0.1, 0.1, 0.06]

    X = np.zeros((batch_size, img_size, img_size))
    Y = np.zeros((batch_size, 1))

    if (output_path != None) and (not os.path.exists(output_path)):
        os.makedirs(output_path)

    for i in range(batch_size):
        y = np.random.randint(len(char_list))
        char = char_list[y]
        angle = max_angle * (np.random.uniform() * 2.0 - 1.0)
        
        back_mean = np.random.randint(min_back_mean, 256)
        back_var_ratio = np.random.uniform() * max_back_var_ratio
        back_var = back_mean * back_var_ratio

        fore_mean = np.random.randint(0, max_fore_mean)
        if (back_mean - fore_mean < 30):
            fore_mean = back_mean / 2
        fore_var_ratio = np.random.uniform() * max_fore_var_ratio
        fore_var = fore_mean * fore_var_ratio

        font_path = font_list[np.random.randint(len(font_list))]
        margin = np.random.choice(margin_list, size = 4, p = probability).tolist()

        char_img = gen_char_img(char, angle, back_mean, back_var, fore_mean, fore_var, font_path, img_size, margin)
        X[i,:,:] = np.array(char_img)
        Y[i, 0] = y

        if (output_path != None):
            char_img.save(output_path + str(i) + ".png") 

    return X, Y

