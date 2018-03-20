import gym
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from q1_schedule import LinearExploration, LinearSchedule
from q2_linear import Linear

from configs.q6_bonus_question import config


class MyDQN(Linear):
    """
    Going beyond - implement your own Deep Q Network to find the perfect
    balance between depth, complexity, number of parameters, etc.
    You can change the way the q-values are computed, the exploration
    strategy, or the learning rate schedule. You can also create your own
    wrapper of environment and transform your input to something that you
    think we'll help to solve the task. Ideally, your network would run faster
    than DeepMind's and achieve similar performance!

    You can also change the optimizer (by overriding the functions defined
    in TFLinear), or even change the sampling strategy from the replay buffer.

    If you prefer not to build on the current architecture, you're welcome to
    write your own code.

    You may also try more recent approaches, like double Q learning
    (see https://arxiv.org/pdf/1509.06461.pdf) or dueling networks 
    (see https://arxiv.org/abs/1511.06581), but this would be for extra
    extra bonus points.
    """
    
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
        
        # target_q # num states x num actions

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

    '''
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
        TODO: implement the computation of Q values like in the paper
                    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

        HINT: you may find tensorflow.contrib.layers useful (imported)
              make sure to understand the use of the scope param

              you can use any other methods from tensorflow
              you are not allowed to import extra packages (like keras,
              lasagne, cafe, etc.)
        """
        ##############################################################
        ################ YOUR CODE HERE - 10-15 lines ################ 
        
        pass

        ##############################################################
        ######################## END YOUR CODE #######################
        return out
    '''


"""
Use a different architecture for the Atari game. Please report the final result.
Feel free to change the configuration. If so, please report your hyperparameters.
"""
if __name__ == '__main__':
    # make env
    env = gym.make(config.env_name)
    env = MaxAndSkipEnv(env, skip=config.skip_frame)
    env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1), 
                        overwrite_render=config.overwrite_render)

    # exploration strategy
    # you may want to modify this schedule
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # you may want to modify this schedule
    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = MyDQN(env, config)
    model.run(exp_schedule, lr_schedule)
