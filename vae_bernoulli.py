from __future__ import division
from __future__ import print_function
import os.path

import tensorflow as tf
from mnist import mnist

class bernoulli_vae():
  def __init__(self, dataset = mnist(), input_dim = 784, encoder_first_layer_dim = 400, decoder_first_layer_dim = 400, latent_dim = 20):
    tf.reset_default_graph()
    self._dataset = dataset
    self._input_dim = input_dim
    self._encoder_first_layer_dim = encoder_first_layer_dim
    self._decoder_first_layer_dim = decoder_first_layer_dim
    self._latent_dim = latent_dim
    self._lam = 0.0
    self.l2_loss = tf.constant(0.0)

  def weight_variable(self, shape):
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial, name = "weights")

  def bias_variable(self, shape):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial, name = "biases")
  def build_linear_layer(self, name_scope, input_dim, output_dim, input):
    with tf.name_scope(name_scope):
      W = self.weight_variable([input_dim, output_dim])
      b = self.bias_variable([output_dim])
      self.l2_loss += tf.nn.l2_loss(W)
      return tf.matmul(input, W) + b
  def gumble(self, shape, eps = 1e-20):
    uniform_samples = tf.random_uniform(shape, name='uniform_samples')
    return -tf.log(-tf.log(uniform_samples + eps) + eps)
  def gumble_softmax(self, logits, name_scope):
    with tf.name_scope(name_scope):
      gumble_samples = self.gumble(tf.shape(logits))
      return tf.reshape(tf.nn.softmax(tf.multiply(gumble_samples + logits, self.beta, name = "gumble_softmax")), [-1, self._latent_dim * 2], 
                        name = 'z')

  def build_encoder_graph(self):
    self.x = tf.placeholder("float", shape=[None, self._input_dim], name = "x")
    tf.add_to_collection('x', self.x)
    encoder_first_layer = tf.nn.relu(self.build_linear_layer(name_scope = 'encoder_first_layer', input_dim = self._input_dim, 
                                                                  output_dim = self._encoder_first_layer_dim, input = self.x))
    self.encoder_logits = tf.reshape(self.build_linear_layer(name_scope = 'encoder_latent_logits', input_dim = self._encoder_first_layer_dim, 
                                                             output_dim = self._latent_dim * 2, input = encoder_first_layer), [-1, 2])
    self.encoder_logits_softmax = tf.nn.softmax(self.encoder_logits)
    tf.add_to_collection('encoder_softmax', tf.reshape(self.encoder_logits_softmax, [-1, self._latent_dim * 2]))
  def export_encoder_meta_graph(self, filename_meta_graph = "models/bernoulli/encoder/model"):
    """You need to call this method directly without calling any other methods that might add any nodes to the tensor graph. In other words, 
    you will need to instantiate a new object from this class and immediately call this method. This method will save the meta graph corresponding 
    to the encoder."""
    if not os.path.exists(os.path.dirname(filename_meta_graph)):
      os.makedirs(os.path.dirname(filename_meta_graph))
    self.build_encoder_graph()
    saver = tf.train.Saver()
    tf.train.Saver.export_meta_graph(saver, filename = filename_meta_graph)

  def build_decoder_graph_stand_alone(self):
    """This is a TF graph that will be used as a stand-along model for decoder and it should not be used to build the TF graph for autoencoder. 
    The main difference between this graph and the one used as a part of autoencoder is the latent variable z. In this graph, z is simply the 
    sample from Bernoulli distribution whereas in autoencoder, z are samples from the gumble-softmax distribution."""
    bernoulli = tf.contrib.distrubtion.Bernoulli(probs = [0.5] * self._laten_dim)
    num_samples = tf.placeholder(tf.int32, shape = [])
    tf.add_to_collection('num_samples', num_samples)
    samples = bernoulli.sample(num_samples)
    samples_negate = 1 - samples
    Samples = tf.stack((samples, samples_negate), axis = 2)
    self.z = tf.reshape(Samples, [num_samples, self._laten_dim * 2])
    
    self.decoder_first_layer = tf.nn.relu(self.build_linear_layer(name_scope = 'decoder_first_layer', input_dim = self._latent_dim * 2, 
                                                                  output_dim = self._decoder_first_layer_dim, input = self.z))
    self.x_hat = self.build_linear_layer(name_scope = 'x_hat', input_dim = self._decoder_first_layer_dim, 
                                         output_dim = self._input_dim, input = self.decoder_first_layer)
    #x_hat are logits. It means that when reconstructio of x, you need to apply tf.nn.sigmoid to x_hat
    tf.add_to_collection('output', f.nn.sigmoid(self.x_hat))
  def build_decoder_graph(self):
    self.decoder_first_layer = tf.nn.relu(self.build_linear_layer(name_scope = 'decoder_first_layer', input_dim = self._latent_dim * 2, 
                                                                  output_dim = self._decoder_first_layer_dim, input = self.z))
    self.x_hat = self.build_linear_layer(name_scope = 'x_hat', input_dim = self._decoder_first_layer_dim, 
                                         output_dim = self._input_dim, input = self.decoder_first_layer)
    #x_hat are logits. It means that when reconstructio of x, you need to apply tf.nn.sigmoid to x_hat

  def build_vae_graph(self):

    self.build_encoder_graph()

    self.beta = tf.placeholder("float", name = "beta")
    self.z = self.gumble_softmax(self.encoder_logits, 'latent_variables')
    
    self.build_decoder_graph()

    self.create_loss_node()


  def create_loss_node(self, eps = 1e-20):
    KLD = tf.reduce_sum(tf.reshape(self.encoder_logits_softmax * tf.log(self.encoder_logits_softmax * 2 + eps), 
                                        [-1, self._latent_dim * 2]), reduction_indices = 1)

    BCE = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.x_hat, labels = self.x), reduction_indices=1)
    #loss = tf.reduce_mean(BCE + KLD)
    self.loss = tf.reduce_mean(BCE + KLD)
    self.BCE_mean = tf.reduce_mean(BCE)
    self.KLD_mean = tf.reduce_mean(KLD)

    #Above, BCE is the negate of the decoder likelihood since it is the cross-entropy between the data x and logits x_hat.
    #So, we would like to 
    #minimize BCE. KLD is actually the KL divergence and we would like to minimize it as well.
    regularized_loss = self.loss + self._lam * self.l2_loss
    loss_summ = tf.summary.scalar("lowerbound", self.loss)
    self.train_step = tf.train.AdamOptimizer(0.01).minimize(regularized_loss)

  def train(self, n_steps = int(1e6), batch_size = 100, path_loss_curve = '', path_model = 'models/bernoulli/vae/model'):
    import os
    if not os.path.exists(os.path.dirname(path_model)):
      os.makedirs(os.path.dirname(path_model))
    if path_loss_curve != '':
      if not os.path.exists(os.path.dirname(path_loss_curve)):
        os.makedirs(os.path.dirname(path_loss_curve))
    else:
      path_loss_curve = os.path.dirname(path_model)
    # add op for merging summary
    summary_op = tf.summary.merge_all()
    # add Saver ops
    saver = tf.train.Saver()
    #mnist.train consists of 55000 images. So, n_steps of 1e6 with batch_size of 100, iterates over each train point about 2000 times.
    #So, in order to only hit any train image only once, for the batch_size of 100, n_steps should be equla to 55000 / 100 = 5500.
    with tf.Session() as sess:
      summary_writer = tf.summary.FileWriter(path_loss_curve, graph=sess.graph)
      sess.run(tf.global_variables_initializer())
      for step in range(1, n_steps):
        batch = self._dataset.train.generate_samples(batch_size)
        feed_dict = {self.x: batch[0], self.beta: 0.01}
        cur_BCE, cur_KLD, _, cur_loss, summary_str = sess.run([self.BCE_mean, self.KLD_mean, self.train_step, self.loss, summary_op], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        
        if step % 50 == 0:
          save_path = saver.save(sess, path_model)
          print("Step {0} | Total Loss: {1} | BCE: {2} | KLD: {3}".format(step, cur_loss, cur_BCE, cur_KLD))

def train_vae(path_model = 'models/bernoulli/vae/model'):
  _bernoulli_vae = bernoulli_vae()
  _bernoulli_vae.build_vae_graph()
  _bernoulli_vae.train(path_model = path_model)

def export_encoder_meta_graph():
  _bernoulli_vae = bernoulli_vae()
  _bernoulli_vae.export_encoder_meta_graph()

def reconstruct_encoder_meta_graph(filename_meta_graph = 'models/bernoulli/encoder/model', path_model = 'models/bernoulli/vae/model'):
    saver = tf.train.import_meta_graph(filename_meta_graph)
    with tf.Session() as sess:
      saver.restore(sess, path_model)
      x = tf.get_collection('x')
      z = tf.get_collection('encoder_softmax')
      print(x)
      print(z)
if __name__ == '__main__':
  #export_encoder_meta_graph()
  #reconstruct_encoder_meta_graph()
  train_vae(path_model = 'models/bernoulli/vae_with_kld/model')

