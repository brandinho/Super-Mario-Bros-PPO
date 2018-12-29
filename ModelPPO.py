import numpy as np
import tensorflow as tf

class PPO():
    def __init__(self, sess, image_height, image_width, image_channels, output_dimension, timesteps, _clip_value, _lr, _gamma, _lambda):
        
        self.sess = sess
        self.image_height = image_height
        self.image_width = image_width
        self.image_channels = image_channels
        self.output_dimension = output_dimension
        
        self.timesteps = timesteps
        
        self._clip_value = _clip_value
        self._lr = _lr
        self._gamma = _gamma
        self._lambda = _lambda
        
        self.c1 = 1
        self.c2 = 0.5
        
        self.kernel_size = [7, 5]
        self.n_filters = [18, 36]
        self.n_strides = [3, 2]
        self.n_hidden = 50
        
        def _build_network(self, name, trainable):
            with tf.variable_scope(name):
                # Convolutional Layers
                conv1 = tf.layers.conv2d(inputs = self.inputs, filters = self.n_filters[0], kernel_size = self.kernel_size[0],
                                         strides = [self.n_strides[0], self.n_strides[0]], padding = "valid", activation = tf.nn.elu, trainable = trainable)
                conv2 = tf.layers.conv2d(inputs = conv1, filters = self.n_filters[1], kernel_size = self.kernel_size[1],
                                         strides = [self.n_strides[1], self.n_strides[1]], padding = "valid", activation = tf.nn.elu, trainable = trainable)

                # Flatten the last Convolutional Layer
                first_dimension = round((((self.image_height - self.kernel_size[0] + 1) / self.n_strides[0]) - self.kernel_size[1] + 1) / self.n_strides[1])
                second_dimension = round((((self.image_width - self.kernel_size[0] + 1) / self.n_strides[0]) - self.kernel_size[1] + 1) / self.n_strides[1])
                dimensionality = first_dimension * second_dimension * self.n_filters[1]
                conv2_flat = tf.reshape(conv2, [-1, dimensionality])

                # Feed into a Dense Layer
                weights_hidden = tf.Variable(tf.random_normal([dimensionality, self.n_hidden], stddev = tf.sqrt(2/(dimensionality + self.n_hidden))), trainable = trainable, name = "WeightsHidden")
                bias_hidden = tf.Variable(tf.zeros([1,self.n_hidden]) + 0.01, trainable = trainable, name = "BiasHidden")
                hidden_layer = tf.nn.elu(tf.matmul(conv2_flat, weights_hidden) + bias_hidden, name = "HiddenLayer")
                
                # Concatenate the hidden layer and the one hot encodings for the previous actions the agent took
                concat_hidden_layer = tf.concat([hidden_layer, self.previous_actions], 1)
                
                # Use Batch Normalization on the Hidden Layer
                hidden_layer_normalized = tf.layers.batch_normalization(concat_hidden_layer, training = self.training, trainable = trainable)                

                # Policy Output
                weights_policy = tf.Variable(tf.random_normal([self.n_hidden + self.timesteps * output_dimension, output_dimension], stddev = tf.sqrt(2/(self.n_hidden + self.timesteps * output_dimension + output_dimension))), trainable = trainable, name = "WeightsPolicy")
                bias_policy = tf.Variable(tf.zeros([1, output_dimension]) + 0.1, trainable = trainable, name = "BiasPolicy")
                policy = tf.nn.softmax(tf.matmul(hidden_layer_normalized, weights_policy) + bias_policy, name = "Policy")

                # Value Output
                weights_value = tf.Variable(tf.random_normal([self.n_hidden + self.timesteps * output_dimension, 1], stddev = tf.sqrt(2/(self.n_hidden + self.timesteps * output_dimension + output_dimension))), trainable = trainable, name = "WeightsValue")
                bias_value = tf.Variable(tf.zeros([1, 1]) + 0.1, trainable = trainable, name = "BiasValue")
                value = tf.add(tf.matmul(hidden_layer_normalized, weights_value), bias_value, name = "Value")
                
            parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = name)                    
            return policy, value, parameters        

        # Input placeholders for the whole model
        self.inputs = tf.placeholder(shape = [None, self.image_height, self.image_width, self.image_channels], dtype = tf.float32, name = "StateInput")
        self.previous_actions = tf.placeholder(shape = [None, self.timesteps * output_dimension], dtype = tf.float32)
        self.training = tf.placeholder(shape = None, dtype = tf.bool)
        
        # Placeholders for Actor optimization
        self.actions = tf.placeholder(shape = [None], dtype = tf.int32, name = "Actions")
        self.GAE = tf.placeholder(shape = [None], dtype = tf.float32, name = "GAE")
        
        # Placeholders for Critic optimization
        self.rewards = tf.placeholder(shape = [None], dtype = tf.float32, name = "Rewards")
        self.next_state_value = tf.placeholder(shape = [None], dtype = tf.float32, name = "NextStateValue")
        
        # Getting the main agent network and the "old" agent network which is used in PPO optimization
        self.policy, self.value, self.parameters = _build_network(self, 'Actor', trainable = True)
        self.old_policy, self.old_value, self.old_parameters = _build_network(self, 'OldActor', trainable = False)
        
        # Update the old network parameters to match the new ones
        with tf.variable_scope('update_old_functions'):
            self.update_old_network = [old_params.assign(params) for params, old_params in zip(self.parameters, self.old_parameters)]
        
        # Define the losses for the actor and the critic (an additional entropy term is in there to ensure the actor explores)
        with tf.variable_scope('loss'):
            with tf.variable_scope('actor_loss'):
                action_probabilities = tf.reduce_sum(self.policy * tf.one_hot(indices = self.actions, depth = output_dimension), axis = 1)
                old_action_probabilities = tf.reduce_sum(self.old_policy * tf.one_hot(indices = self.actions, depth = output_dimension), axis = 1)
                
                self.ratios = tf.exp(tf.log(action_probabilities) - tf.log(old_action_probabilities))
                self.clipped_ratios = tf.clip_by_value(self.ratios, clip_value_min = 1 - self._clip_value, clip_value_max = 1 + self._clip_value)
                self.clipped_loss = tf.minimum(tf.multiply(self.GAE, self.ratios), tf.multiply(self.GAE, self.clipped_ratios))
                self.actor_loss = tf.reduce_mean(self.clipped_loss)
                
            with tf.variable_scope('critic_loss'):
                self.critic_loss = tf.reduce_mean(tf.squared_difference(self.rewards + self._gamma * self.next_state_value, self.value))
                
            with tf.variable_scope('entropy'):
                entropy = -tf.reduce_sum(self.policy * tf.log(tf.clip_by_value(self.policy, 1e-10, 1)), axis = 1)
                self.entropy = tf.reduce_mean(entropy, axis = 0)
                
            self.loss = - (self.actor_loss - self.c1 * self.critic_loss + self.c2 * self.entropy)

        # Perform Gradient Descent to train the model
        with tf.variable_scope('trainModel'):
            optimizer = tf.train.AdamOptimizer(self._lr)
            self.train_op = optimizer.minimize(self.loss, var_list = self.parameters)
    
    # Training function that uses mini-batches that are shuffled between each epoch
    def train(self, inputs, actions, GAE, rewards, next_state_value, prev_actions_list, epochs, batch_size):
        batches_per_epoch = inputs.shape[0]//batch_size + 1
        
        for i in range(epochs):
            shuffled_indexes = np.random.choice(inputs.shape[0], size = inputs.shape[0], replace = False)
            batch_num = 0
            for j in range(batches_per_epoch):
                current_index = shuffled_indexes[batch_num:(batch_num + batch_size)]
                
                self.sess.run(self.train_op, {self.inputs: inputs[current_index,], self.actions: actions[current_index,], 
                                              self.GAE: GAE[current_index,], self.rewards: rewards[current_index,], 
                                              self.next_state_value: next_state_value[current_index,], self.training: True,
                                              self.previous_actions: prev_actions_list[current_index,]})
                
                batch_num += batch_size

    def update(self):
        self.sess.run(self.update_old_network)
