import numpy as np
from keras.models import Sequential
from keras.layers import ConvLSTM2D
from keras.optimizers import SGD
from keras.engine.topology import Layer
from keras import backend as K
import random


class softmax_prime(Layer):
    def call(self, x):
        x1 = K.exp(x[0, 0, :, :]) / K.sum(K.exp(x[0, 0, :, :]))
        x2 = x[0, 1, :, :]
        y = K.expand_dims(K.stack((x1, x2)), axis=0)
        return y


class PGAgent:
    def __init__(self, dim_featuremaps):
        """
        Parameters:
            state_size: shape of state is (dim, dim), so it's dim * dim
            action_size: same as state_size
            gamma: the one in Bellman Equation
            learning_rate: learning rate in RMS
            n_filters: number of filters in deep Q network
            connect_thr: threadhold of connecting two node, the value is in probs matrix
            remove_thr: similar to connect_thr, but it indicates to remove the max_connection
            states: a square matrix
            probs: a square matrix indicates the posterior probability of the relation between
                   any two different bits. Sum of all is 1
        """
        self.gamma = 0.99
        self.learning_rate = 2e-5
        self.connect_thr = 0
        self.remove_thr = 0

        self.dim_featuremaps = dim_featuremaps
        self.inputs = []
        self.target_depends = []
        self.target_influences = []
        self.depend_grads1 = []
        self.depend_grads2 = []
        self.influence_grads = []
        self.rewards1 = []
        self.rewards2 = []
        self.rewards3 = []

        self.epsilon = 0.05
        self.model = self._build_model()
        # self.model.summary()

    def _build_model(self):
        model = Sequential()
        model.add(ConvLSTM2D(2, (1, 1), border_mode='same', init='he_uniform',
                             input_shape=(
                             len(self.dim_featuremaps), 2, max(self.dim_featuremaps), max(self.dim_featuremaps)),
                             data_format='channels_first',
                             return_sequences=True))

        opt = SGD(nesterov=True, lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    def remember(self, last_dependency_state, last_influence_state,
                 next_dependency_state, next_influence_state_con, next_influence_state_dis,
                 action1, action2,
                 reward1, reward2, reward3):
        # remember data: time * channels * rows * cols
        time_states = []
        for i in range(len(self.dim_featuremaps)):
            time_states.append(np.vstack([[last_dependency_state[i]], [last_influence_state[i]]]))
        self.inputs.append(np.vstack([time_states]))

        # remember gradient & reward: time * rows * cols
        time_depend_grad1, time_depend_grad2, time_influence_grad = [], [], []
        time_reward1, time_reward2, time_reward3 = [], [], []
        for i in range(len(self.dim_featuremaps)):
            if action1[i] > -1:
                temp = np.zeros([max(self.dim_featuremaps), max(self.dim_featuremaps)])
                temp[int(action1[i] / self.dim_featuremaps[i]), action1[i] % self.dim_featuremaps[i]] = 1
                time_depend_grad1.append(np.array(temp).astype('float32') - next_dependency_state[i])
                time_reward1.append(reward1[i])
            else:
                time_depend_grad1.append(
                    np.array(np.zeros([max(self.dim_featuremaps), max(self.dim_featuremaps)])).astype('float32'))
                time_reward1.append(0)

            if action2[i] > -1:
                temp = np.zeros([max(self.dim_featuremaps), max(self.dim_featuremaps)])
                temp[int(action2[i] / self.dim_featuremaps[i]), action2[i] % self.dim_featuremaps[i]] = 1
                time_depend_grad2.append(-(np.array(temp).astype('float32') - np.log(1 - next_dependency_state[i])) / (
                            1 - next_dependency_state[i]))
                time_reward2.append(reward2[i])
            else:
                time_depend_grad2.append(
                    np.array(np.zeros([max(self.dim_featuremaps), max(self.dim_featuremaps)])).astype('float32'))
                time_reward2.append(0)

            time_influence_grad.append(next_influence_state_dis[i] - next_influence_state_con[i])
        self.depend_grads1.append(np.vstack([time_depend_grad1]))
        self.depend_grads2.append(np.vstack([time_depend_grad2]))
        self.influence_grads.append(np.vstack([time_influence_grad]))
        self.rewards1.append(np.vstack([time_reward1]))
        self.rewards2.append(np.vstack([time_reward2]))
        self.rewards3.append(np.vstack([reward3]))
        self.target_depends.append(next_dependency_state)
        self.target_influences.append(next_influence_state_con)

    # input:batch * time * channels * rows * cols
    def act(self, input):
        action1 = [-1] * len(self.dim_featuremaps)
        action2 = [-1] * len(self.dim_featuremaps)
        next_dependency_state = []
        next_influence_state = []

        ''' 
            init_prob_matrix: the output of the network, indicating the prob of the action(maximum)
            log_prob_matrix: the minus log of the init_prob_matrix, also indicating the prob of the action, but is of reverse order(minimum)
        '''
        prediction = self.model.predict(input, batch_size=1)
        next_dependency_state = prediction[0, :, 0, :, :]
        next_influence_state = prediction[0, :, 1, :, :]

        # get action1, action2
        for i in range(len(self.dim_featuremaps)):
            sim_next_dependency_state = next_dependency_state[i]
            if self.dim_featuremaps[i] < max(self.dim_featuremaps):
                sim_next_dependency_state = next_dependency_state[i][:self.dim_featuremaps[i], :self.dim_featuremaps[i]]

            if np.sum(np.sum(sim_next_dependency_state)) != 0:
                prob = sim_next_dependency_state / np.sum(np.sum(sim_next_dependency_state))
            else:
                prob = np.ones((1, self.dim_featuremaps[i], self.dim_featuremaps[i])) / (
                            self.dim_featuremaps[i] * self.dim_featuremaps[i])

            # reshape into 1-D array
            prob_trans = prob.reshape((self.dim_featuremaps[i] * self.dim_featuremaps[i]))

            # get action1
            if max(prob_trans) > self.connect_thr:
                random_num = random.random()
                # epsilon-greedy
                if random_num > self.epsilon:
                    action1[i] = np.where(prob_trans == np.max(prob_trans))[0][0]
                else:
                    action1[i] = np.random.choice(self.dim_featuremaps[i] * self.dim_featuremaps[i], 1)[0]

            # get action2
            if min(prob_trans) < self.remove_thr:
                random_num = random.random()
                if random_num > self.epsilon:
                    action2[i] = np.where(prob_trans == np.min(prob_trans))[0][0]
                else:
                    action2[i] = np.random.choice(self.dim_featuremaps[i] * self.dim_featuremaps[i], 1)[0]

        return action1, action2, next_dependency_state, next_influence_state

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in range(0, rewards.size):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train(self):
        # get inputs
        X = np.vstack([self.inputs])
        # get outputs
        # process gradients: batch * time * rows * cols
        batch_target = []
        time_target = []
        for i in range(len(self.depend_grads1)):
            time_target = []
            for j in range(self.depend_grads2[0].shape[0]):
                rewards1 = self.discount_rewards(np.vstack(self.rewards1)[:, j])
                rewards2 = self.discount_rewards(np.vstack(self.rewards2)[:, j])
                rewards3 = self.discount_rewards(np.vstack(self.rewards3)[:, j])
                target_depends = self.target_depends[i][j] + self.depend_grads1[i][j] * rewards1[i] + \
                                 self.depend_grads2[i][j] * rewards2[i]
                target_influence = self.target_influences[i][j] + self.influence_grads[i][j] * rewards3[i]
                target = np.vstack([[target_depends], [target_influence]])
                time_target.append(target)
            batch_target.append(np.vstack([time_target]))

        Y = np.vstack([batch_target])

        self.model.train_on_batch(X, Y)
        self.inputs, self.target_depends, self.target_influences = [], [], []
        self.depend_grads1, self.depend_grads2, self.influence_grads = [], [], []
        self.rewards1, self.rewards2, self.rewards3 = [], [], []

    def save(self, name):
        self.model.save_weights(name)
