import os
import numpy as np
from skimage.io import imsave

class STL10:
    def __init__(self, img_dir='stl10'):
        self.img_dir = img_dir
        self.bin_dir = 'stl10_binary'

    def get_files(self, target):
        assert target in ["train", "test", "unlabeled"]
        if target in ["train", "test"]:
            images = self.load_images(os.path.join(self.bin_dir, target+"_X.bin"))
            labels = self.load_labels(os.path.join(self.bin_dir, target+"_y.bin"))
        else:
            images = self.load_images(os.path.join(self.bin_dir, target+"_X.bin"))
            labels = None
        return images, labels

    def load_images(self, image_binary):
        with open(image_binary, "rb") as fp:
            images = np.fromfile(fp, dtype=np.uint8)
            images = images.reshape(-1, 3, 96, 96)
            return np.transpose(images, (0, 3, 2, 1))

    def load_labels(self, label_binary):
        with open(label_binary) as fp:
            labels = np.fromfile(fp, dtype=np.uint8)
            return labels.reshape(-1, 1) - 1 # 1-10 -> 0-9

    def save_images_and_labels(self, images, labels, mode):
        """modeは train, test のいずれか"""
        assert mode in ["train", "test"]
        img_dir = f'{self.img_dir}/{mode}'
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)

        with open(f'{img_dir}.txt', 'a') as f:
            for i, img in enumerate(images):
                imsave(f'{img_dir}/{i}.png', img)
                f.write(f'{img_dir}/{i}.png,{labels[i].flatten()}\n')

def stl10_converter():
    stl10 = STL10('./stl10')
    train_X, train_y = stl10.get_files("train")
    test_X, test_y   = stl10.get_files("test")
    #unlabeld_X,_ = stl10.get_files("unlabeld")
    
    stl10.save_images_and_labels(train_X, train_y, 'train')
    stl10.save_images_and_labels(test_X, test_y, 'test')
    #stl10.save_images_and_labels(unlabeld_X,_, 'unlabeld')
