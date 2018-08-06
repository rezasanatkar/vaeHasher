from __future__ import print_function
import tensorflow as tf
class meta_graph_utils():
    @staticmethod
    def examine_checkpoint_file(model_path = './model'):
        """This method prints the variable names stored in the check point file matching the model_path."""
        from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
        print_tensors_in_checkpoint_file(model_path, tensor_name = '', all_tensors = False)

    @staticmethod
    def fetch_node_from_meta_graph(node_name, filename_meta_graph = 'models/bernoulli/encoder/model'):
        """This method fetch the tensor node with the node_name and return its corresponding node in the metagraph."""
        saver = tf.train.import_meta_graph(filename_meta_graph)
        nodes = tf.get_collection(node_name)
        return nodes

