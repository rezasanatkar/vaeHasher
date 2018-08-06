from __future__ import print_function
import tensorflow as tf
class encoder(object):
    """This main functionality of this class is to encode the mnist images into a tuple of mean and logvar of a gaussin approximate 
    posterior learned by VAE encoder in vae.py."""
    def __init__(self, filename_meta_graph = 'models/bernoulli/encoder/model', path_model = 'models/bernoulli/vae/model', input_name = 'x',
                 output_name = 'encoder_softmax'):
        saver = tf.train.import_meta_graph(filename_meta_graph)
        self._sess = tf.Session()
        saver.restore(self._sess, path_model)
        self._input = tf.get_collection(input_name)[0]
        self._output = tf.get_collection(output_name)[0]
            #[0] in above is that get_collection will return a list like [input]
    def encode_samples(self, x):
        """This method needs to be called to encode the input images. x has to be a numpy array of the shape [num_images, 784].
        This method returns the mean and logvar of the approximated gaussian posterior."""
        return self._sess.run([self._output], feed_dict = {self._input: x})
    def compute_codes(self, x):
        """This method needs to be called to encode the input images. x has to be a numpy array of the shape [num_images, 784].
        This method returns the mean the approximated gaussian posterior."""
        return self.encode_samples(x)[0]
    def __del__(self):
        self._sess.close()
def main():
    _encoder = encoder()
    from mnist import mnist
    _mnist = mnist()
    samples = _mnist.train.generate_samples(1)[0]
    print(_encoder.encode_samples(samples))
                          
if __name__ == "__main__":
    main()
