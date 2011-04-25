#!/usr/bin/env python

# Experiments with blocked EM inference.
# Daniel Klein, 4/25/2011

# By default (controlled by preprocess flag), the raw images are
# inverted and thresholded.

label_file = 'digits/train-labels-idx1-ubyte'
image_file = 'digits/train-images-idx3-ubyte'

def get_data(how_many, preprocess = True):
    labels = []
    images = []

    with open(label_file, 'rb') as label:
        header = label.read(8)
        for i in range(how_many):
            labels.append(ord(label.read(1)))

    with open(image_file, 'rb') as image:
        header = image.read(16)
        for i in range(how_many):
            raw = map(ord, image.read(28 * 28))
            if preprocess:
                processed = map(lambda x: (x <= 128 and 255 or 0), raw)
                images.append(processed)
            else:
                images.append(raw)

    return zip(labels, images)

if __name__ == '__main__':
    how_many = 20
    examples = get_data(how_many)

    from PIL import Image, ImageDraw
    im = Image.new('L', (how_many * 28, 46), 255)

    offset = 0
    for l, i in examples:
        digit = Image.new('L', (28, 28))
        digit.putdata(i)
        im.paste(digit, (offset, 0))
        draw = ImageDraw.Draw(im)
        draw.text((offset + 10, 31), str(l))
        del draw
        offset += 28
    im.show()
