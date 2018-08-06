from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
class dataset(object):
    def __init__(self, source):
        self._source = source
        self._epoch_completed = False
    @property
    def epoch_completed(self):
        """It will be True if every image in the dataset has been visited at least once."""
        return self._epoch_completed
    def generate_samples(self, num_samples):
        """This method will return tuple of images and their labels. Eg, ([[image1], [image2]], [label1 label2]), such that
        in the first element of the tuple, every row corresponds to one image."""
        samples = self._source.next_batch(num_samples)
        self._epoch_completed = self._source.epochs_completed != 0
        return samples
class mnist(object):
    """This class main functionality is to draw mnist images and generate mnist samples using tensorflow train module."""
    def __init__(self):
        self._mnist = input_data.read_data_sets('MNIST') 
        self.train = dataset(self._mnist.train)
        self.test = dataset(self._mnist.test)
        self.validation = dataset(self._mnist.validation)
    def draw_image(self, image, threshold = 0.5):
        """This method draw mnist images represented by image numpy array of form [num_images, 28 * 28].
        the image also could be a tuple of images and labels where images are the the first element the tuple, and is 
        described as above.
        threshold denote the threshold from (0,1) that acts as the decision criteria for drawing binary images."""
        if type(image) is tuple:
            image = image[0]
        for i in range(image.shape[0]):
            for j in range(28):
                for k in range(28):
                    if image[i, j * 28 + k] > threshold:
                        print('*', end = '')
                    else:
                        print(' ', end = '')
                print('\n')
                    
def main():
    _mnist = mnist()
    _mnist.draw_image(_mnist.validation.generate_samples(2))
if __name__ == "__main__":
    main()
