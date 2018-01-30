from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym, gym.spaces
from baselines.common.distributions import make_pdtype


class TransferMlpPolicy:
    recurrent = False

    def __init__(self, name, ob_space, ac_space,
                 src_ob_space, hid_size, num_hid_layers):
        assert isinstance(src_ob_space, gym.spaces.Box)
        self.num_hid_layers = num_hid_layers
        self.hid_size = hid_size
        self.scope = tf.get_variable_scope().name
        self.src_ob_space = src_ob_space

        with tf.variable_scope(name):
            self._init(ob_space, ac_space)

    def _init(self, ob_space, ac_space, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        src_ob = U.get_placeholder(name="ob",
                                   dtype=tf.float32,
                                   shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        obz = tf.clip_by_value((src_ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

        if self.src_ob_space.shape != ob_space.shape:
            # transfer observation space from ob_space to src_ob_space
            last_out = tf.nn.tanh(U.dense(src_ob,
                                          max(self.src_ob_space.shape),
                                          "transfer_ob",
                                          weight_init=U.normc_initializer(1.0)))
        else:
            last_out = obz

        for i in range(self.num_hid_layers):
            last_out = tf.nn.tanh(
                U.dense(last_out, self.hid_size, "vffc%i" % (i + 1), weight_init=U.normc_initializer(1.0)))
        self.vpred = U.dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:, 0]

        last_out = obz
        for i in range(self.num_hid_layers):
            last_out = tf.nn.tanh(
                U.dense(last_out, self.hid_size, "polfc%i" % (i + 1), weight_init=U.normc_initializer(1.0)))
        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = U.dense(last_out, pdtype.param_shape()[0] // 2, "polfinal", U.normc_initializer(0.01))
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0] // 2],
                                     initializer=tf.zeros_initializer())
            pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)
        else:
            pdparam = U.dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, src_ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []
