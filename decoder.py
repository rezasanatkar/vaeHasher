from __future__ import print_function
import tensorflow as tf
class decoder(object):
    """This class generate samples based on a decoder learned by running vae.py.
       This decoder is a two layer fully connected neural network with the first layer having hidden_dim nodes and the second layer 
        that is in charge of recunstruction of x has input_dim nodes. The first layer's responsibility is to map the random generated gaussian
        with the dimension of latent_dim to a vector with hidden_dim nodes."""

    def __init__(self, input_dim = 784, hidden_dim = 400, latent_dim = 20):
        """input_dim denotes the dimension of the samples that will be reconstructed. 
        hidden_dim is the number of hidden units for the first hidden layer. 
        latent_dim is the dimension of generated random gaussin vector."""
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._latent_dim = latent_dim
        self._build_graph()

    def _build_graph(self):
        """This method needs to be called to build the graph."""
        tf.reset_default_graph()
        self._num_samples = tf.placeholder(tf.int32, shape = [])
        #shape = [] refers to a scalar
        #Sample z
        z = tf.random_normal([self._num_samples, self._latent_dim], name='z')
        
        with tf.name_scope("hidden_decoder"):
            # Hidden layer decoder
            W_z_hidden = self._weight_variable([self._latent_dim, self._hidden_dim])
            b_z_hidden = self._bias_variable([self._hidden_dim])
            hidden = tf.nn.relu(tf.matmul(z, W_z_hidden) + b_z_hidden)

        with tf.name_scope("x_hat"):        
            # Constructing the input X
            W_hidden_reconstruction = self._weight_variable([self._hidden_dim, self._input_dim])
            b_hidden_reconstruction = self._bias_variable([self._input_dim])
            self._x_hat = tf.nn.sigmoid(tf.matmul(hidden, W_hidden_reconstruction) + b_hidden_reconstruction, name = "x_hat")

    def _weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.001)
        return tf.Variable(initial, name = "weights")

    def _bias_variable(self, shape):
        initial = tf.constant(0., shape=shape)
        return tf.Variable(initial, name = "biases")

    def generate_samples(self, num_samples, model_path = "save/model.ckpt"):
        """This method generates the num_samples number of samples."""
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, model_path)
            return sess.run(self._x_hat, feed_dict = {self._num_samples: num_samples})

def main():
    _decoder = decoder()
    from mnist import mnist
    _mnist = mnist()
    temp = _decoder.generate_samples(20)
    _mnist.draw_image(temp, threshold = 0.5)
                          
if __name__ == "__main__":
    main()
