import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from core.deep_q_learning import DQN
from q1_schedule import LinearExploration, LinearSchedule

from configs.q2_linear import config


class Linear(DQN):
    """
    Implement Fully Connected with Tensorflow
    """
    def add_placeholders_op(self):
        """
        Adds placeholders to the graph

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model
        """
        # this information might be useful
        # here, typically, a state shape is (80, 80, 1)
        state_shape = list(self.env.observation_space.shape)

        ##############################################################
        """
        TODO: add placeholders:
              Remember that we stack 4 consecutive frames together, ending up with an input of shape
              (80, 80, 4).
               - self.s: batch of states, type = uint8
                         shape = (batch_size, img height, img width, nchannels x config.state_history)
               - self.a: batch of actions, type = int32
                         shape = (batch_size)
               - self.r: batch of rewards, type = float32
                         shape = (batch_size)
               - self.sp: batch of next states, type = uint8
                         shape = (batch_size, img height, img width, nchannels x config.state_history)
               - self.done_mask: batch of done, type = bool
                         shape = (batch_size)
                         note that this placeholder contains bool = True only if we are done in 
                         the relevant transition
               - self.lr: learning rate, type = float32
        
        (Don't change the variable names!)
        
        HINT: variables from config are accessible with self.config.variable_name
              Also, you may want to use a dynamic dimension for the batch dimension.
              Check the use of None for tensorflow placeholders.

              you can also use the state_shape computed above.
        """
        ##############################################################
        ################YOUR CODE HERE (6-15 lines) ##################
        self.s = tf.placeholder(tf.uint8, (None, state_shape[0], state_shape[1], state_shape[2]*4)) # only 1 channel after preprocess
        self.a = tf.placeholder(tf.int32, (None))
        self.r = tf.placeholder(tf.float32, (None))
        self.sp = tf.placeholder(tf.uint8, (None, state_shape[0], state_shape[1], state_shape[2]*4)) # only 1 channel after preprocess
        self.done_mask = tf.placeholder(tf.bool, (None))
        self.lr = tf.placeholder(tf.float32, shape=[]) # self.config.lr?
        ##############################################################
        ######################## END YOUR CODE #######################


    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n
        out = state

        ##############################################################
        """
        TODO: implement a fully connected with no hidden layer (linear
            approximation) using tensorflow. In other words, if your state s
            has a flattened shape of n, and you have m actions, the result of 
            your computation sould be equal to
                s * W + b where W is a matrix of shape n x m and b is 
                a vector of size m (you should use bias)

        HINT: you may find tensorflow.contrib.layers useful (imported)
              make sure to understand the use of the scope param
              make sure to flatten the state input (see tensorflow.contrib.layers.flatten())    

              you can use any other methods from tensorflow
              you are not allowed to import extra packages (like keras,
              lasagne, cafe, etc.)
        """
        ##############################################################
        ################ YOUR CODE HERE - 2-3 lines ################## 
        with tf.variable_scope(scope, reuse=reuse):
            out = tf.contrib.layers.fully_connected(inputs=tf.contrib.layers.flatten(out),
                num_outputs=num_actions, # one q val output for each action
                reuse=reuse,
                scope="FCL1",
                activation_fn=None) # linear so no activation fn # reset out
        ##############################################################
        ######################## END YOUR CODE #######################

        return out


    def add_update_target_op(self, q_scope, target_q_scope):
        """
        update_target_op will be called periodically 
        to copy Q network weights to target Q network

        Remember that in DQN, we maintain two identical Q networks with
        2 different set of weights. In tensorflow, we distinguish them
        with two different scopes. One for the target network, one for the
        regular network. If you're not familiar with the scope mechanism
        in tensorflow, read the docs

        Periodically, we need to update all the weights of the Q network 
        and assign them with the values from the regular network. Thus,
        what we need to do is to build a tf op, that, when called, will 
        assign all variables in the target network scope with the values of 
        the corresponding variables of the regular network scope.
    
        Args:
            q_scope: (string) name of the scope of variables for q
            target_q_scope: (string) name of the scope of variables
                        for the target network
        """
        ##############################################################
        """
        TODO: add an operator self.update_target_op that assigns variables
            from target_q_scope with the values of the corresponding var 
            in q_scope

        HINT: you may find the following functions useful:
            - tf.get_collection
            - tf.assign
            - tf.group

        (be sure that you set self.update_target_op)
        """
        ##############################################################
        ################### YOUR CODE HERE - 5-10 lines #############
        opAssigns = []
        q_values = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=q_scope)
        target_q_values = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_q_scope)
        for i in range(len(q_values)):
            opAssigns.append(tf.assign(target_q_values[i], q_values[i])) # name  # networks should have same weights now
        self.update_target_op = tf.group(*opAssigns) 
        ##############################################################
        ######################## END YOUR CODE #######################


    def add_loss_op(self, q, target_q):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
        """
        # you may need this variable
        num_actions = self.env.action_space.n

        ##############################################################
        """
        TODO: The loss for an example is defined as:
                Q_samp(s) = r if done
                          = r + gamma * max_a' Q_target(s', a')
                loss = (Q_samp(s) - Q(s, a))^2 

              You need to compute the average of the loss over the minibatch
              and store the resulting scalar into self.loss

        HINT: - config variables are accessible through self.config
              - you can access placeholders like self.a (for actions)
                self.r (rewards) or self.done_mask for instance
              - target_q is the q-value evaluated at the s' states (the next states)  
              - you may find the following functions useful
                    - tf.cast
                    - tf.reduce_max / reduce_sum
                    - tf.one_hot
                    - ...

        (be sure that you set self.loss)
        """
        ##############################################################
        ##################### YOUR CODE HERE - 4-5 lines #############
        
        '''
        # target_q # num states x num actions
        q_of_optimal_a = tf.reduce_max(target_q, reduction_indices=[1]) # optimal action's val for each state # num states x 1
        
        gamma = tf.constant(self.config.gamma, dtype=tf.float32, name="gamma")
        Q_samp = self.r + tf.cast(tf.logical_not(self.done_mask), tf.float32)*gamma*q_of_optimal_a # num states x 1
        # q # num states x num actions
        actions = tf.one_hot(self.a, num_actions) # matrix num states x num_actions, batch of states -- self.a is the action you took on that particular state
        Q_sa = tf.reduce_sum(q*actions, reduction_indices=[1]) # reduce sum across actions to get num_states x 1
        self.loss = tf.reduce_mean(tf.square(Q_samp - Q_sa)) # SGD loss "squared expectation" (remember which Q network max taken over...)
        '''

        # taking a max over q to select the action index
        # use that index to index into target q and set q_of_optimal_a to that val of target q
        optimal_a = tf.one_hot(tf.argmax(q, axis=1), num_actions) # added # one hot 
        q_of_optimal_a = tf.reduce_sum(tf.multiply(target_q, optimal_a), axis=1) # added 
        # removed # q_of_optimal_a = tf.reduce_max(target_q, reduction_indices=[1]) # optimal action's val for each state # num states x 1
        
        gamma = tf.constant(self.config.gamma, dtype=tf.float32, name="gamma")
        Q_samp = self.r + tf.cast(tf.logical_not(self.done_mask), tf.float32)*gamma*q_of_optimal_a # num states x 1
        # q # num states x num actions
        actions = tf.one_hot(self.a, num_actions) # matrix num states x num_actions, batch of states -- self.a is the action you took on that particular state
        Q_sa = tf.reduce_sum(q*actions, reduction_indices=[1]) # reduce sum across actions to get num_states x 1
        self.loss = tf.reduce_mean(tf.square(Q_samp - Q_sa)) # SGD loss "squared expectation" (remember which Q network max taken over...)

        ##############################################################
        ######################## END YOUR CODE #######################


    def add_optimizer_op(self, scope):
        """
        Set self.train_op and self.grad_norm
        """

        ##############################################################
        """
        TODO: 1. get Adam Optimizer (remember that we defined self.lr in the placeholders
                section)
              2. compute grads wrt to variables in scope for self.loss
              3. clip the grads by norm with self.config.clip_val if self.config.grad_clip
                is True
              4. apply the gradients and store the train op in self.train_op
               (sess.run(train_op) must update the variables)
              5. compute the global norm of the gradients and store this scalar
                in self.grad_norm

        HINT: you may find the following functinos useful
            - tf.get_collection
            - optimizer.compute_gradients
            - tf.clip_by_norm
            - optimizer.apply_gradients
            - tf.global_norm
             
             you can access config variable by writing self.config.variable_name

        (be sure that you set self.train_op and self.grad_norm)
        """
        ##############################################################
        #################### YOUR CODE HERE - 8-12 lines #############

        adam_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope) # onlytarget vars-wrong X , get vars in scope
        grads_and_vars = adam_optimizer.compute_gradients(self.loss, var_list) # [[grad1, name1], [grad2, name2] ... ]
        # gradient clipping
        if self.config.grad_clip:
            for i in range(len(grads_and_vars)):
                print 'grads_and_vars[i][0]: ', grads_and_vars[i][0]
                grads_and_vars[i] = (tf.clip_by_norm(grads_and_vars[i][0], self.config.clip_val), grads_and_vars[i][1])
        grads = [item[0] for item in grads_and_vars] # just get out the grads
        self.train_op = adam_optimizer.apply_gradients(grads_and_vars) # pass in grads
        self.grad_norm = tf.global_norm(grads)
        
        ##############################################################
        ######################## END YOUR CODE #######################
    


if __name__ == '__main__':
    env = EnvTest((5, 5, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = Linear(env, config)
    model.run(exp_schedule, lr_schedule)