from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import time as time
import numpy as np
import tensorflow as tf
import networks
import os


class CNN(object):
    def __init__(self, sess,
                 input_shape = (28, 28, 1),
                 num_classes=10,
                 batch_size=64,
                 learning_rate=0.002,
                 epochs=25,
                 beta=.5,
                 lambda_1=1,
                 lambda_2=1,

                 sumup=False,
                 model_name='CNN',
                 checkpoint_dir="checkpoint",

                 ):

        self.sess = sess



        # Inputs
        self.input_shape = list(input_shape)
        self.num_classes =num_classes

        self.inputs = tf.placeholder(
            tf.float32, [None] + self.input_shape, name='inputs')


        # Generator

        self.adv = self.Generator(self.inputs)
        self.logits_nat =self.ClassifierNetwork(self.inputs,reuse=False)

        self.logits_adv= self.ClassifierNetwork(self.G,reuse=True)

        #Loss

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.epsilon = epsilon

        cross_entropy_nat = tf.reduce_mean(
                                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits_nat,labels=self.labels))
        cross_entropy_adv = tf.reduce_mean(
                                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits_adv,labels=self.labels))
        norm_G = tf.maximum(tf.norm(self.Generator-self.inputs)**2,epsilon) # TODO: reducemean


        self.loss_N = cross_entropy_nat+lambda_1*cross_entropy_adv+lambda_2*norm_G
        self.loss_G = -self.loss_N

        t_vars = tf.trainable_variables()

        self.N_vars = [var for var in t_vars if 'N_' in var.name]
        self.G_vars = [var for var in t_vars if 'G_' in var.name]


        # Training params

        self.batch_size = batch_size
        self.epochs = epochs



        # optim params
        self.learning_rate = learning_rate
        self.beta = beta

        self.optimizer_N = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta) \
            .minimize(self.global_loss, var_list=self.vars)
        self.optimizer_G = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta) \
            .minimize(self.global_loss, var_list=self.vars)





        # Checkpoints

        self.model_name = model
        self.checkpoint_dir = os.path.join(checkpoint_dir,model_name)




        self.saver = tf.train.Saver()




        if not os.path.exists("./logs_train"+model_name):
            os.makedirs("./logs_train"+model_name)
        files = os.listdir("./logs_train"+model_name)
        for file in files:
            os.remove("./logs_train" +model_name +'/' + file)

        if not os.path.exists("./logs_val"+model_name):
            os.makedirs("./logs_val"+model_name)
        files = os.listdir("./logs_val"+model_name)
        for file in files:
            os.remove("./logs_val"+model_name + '/' + file)

        self.writer_train = tf.summary.FileWriter("./logs_train"+model_name, self.sess.graph)
        self.writer_val = tf.summary.FileWriter("./logs_val"+model_name, self.sess.graph)

        if self.lambda_loss==0:
            self.writer_grad_val = tf.summary.FileWriter("./logs_grad" + model_name, self.sess.graph)

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            self.counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            self.counter = 0
            print(" [!] Load failed...")



    def build_model(self):
        self.mode=tf.placeholder(tf.string, name='mode')
        self.noise_type=tf.placeholder(tf.string, name='noise_type')
        self.noise=tf.placeholder(
            tf.float32, [None], name='noise')
        self.inputs = tf.placeholder(
            tf.float32, [None] + self.input_shape, name='inputs')
        inputs = self.inputs

        self.labels = tf.placeholder(
            tf.int64, [None, self.num_labels], name='labels')
        labels = self.labels

        self.network = networks.network_mnist(inputs, self.input_shape, self.num_labels, self.mode, self.noise_type, self.noise)
        self.network_sum = tf.summary.histogram("cnn", self.network)

        self.loss_1 = networks.cross_entropy_loss(self.network, labels)
        self.loss_1_sum = tf.summary.scalar("cross_entropy_loss", self.loss_1)
        self.embed = tf.get_default_graph().get_tensor_by_name("embedding/Relu:0")

        pp = [-1] + self.input_shape + [1]
        self.gradient_embedding = tf.concat([tf.reshape(
            tf.gradients(self.embed[:, i], inputs)[0], pp) for i in range(self.embed.shape[1])], axis=4)
        self.loss_2 = networks.representer_grad_loss(self.gradient_embedding)
        if self.lambda_loss!=0:
            self.loss_2_sum = tf.summary.scalar("representer_grad_loss", self.loss_2)
            self.global_loss = self.loss_1+self.lambda_loss*self.loss_2
        else:
            self.global_loss = self.loss_1
        self.loss_sum = tf.summary.scalar("global_loss", self.global_loss)

        self.vars = tf.trainable_variables()

        self.saver = tf.train.Saver()

        self.acc = networks.accuracy(self.network, self.labels)
        self.acc_sum = tf.summary.scalar("accuracy", self.acc)

        self.summary = tf.summary.merge_all()
        if self.lambda_loss==0:
            self.loss_2_sum = tf.summary.scalar("representer_grad_loss", self.loss_2)

    def train(self, X, y,cv=0.05,noise_type='',noise=np.zeros(1000)):

        self.train_size = np.shape(X)[0]
        tr_set = np.random.choice(np.arange(self.train_size), int(self.train_size*(1-cv)), replace=False)
        Xtr = X[tr_set]
        ytr = y[tr_set]
        Xval = np.delete(X, tr_set, axis=0)
        yval = np.delete(y, tr_set, axis=0)
        self.train_size = np.shape(Xtr)[0]
        counter = self.counter
        start_time = time.time()


        batch_idxs = self.train_size // self.batch_size
        i=0
        for epoch in range(self.counter, self.epoch):

            for idx in range(0, batch_idxs):
                batch_images = (Xtr[idx * self.batch_size:(idx + 1) * self.batch_size]).reshape(
                    tuple([self.batch_size]+self.input_shape))
                batch_labels = (ytr[idx * self.batch_size:(idx + 1) * self.batch_size])

                # Update network
                _, summary_str = self.sess.run([self.optimizer, self.summary],
                                               feed_dict={
                                                   self.inputs: batch_images,
                                                   self.labels: batch_labels,
                                                   self.mode: "TRAIN",
                                                   self.noise_type:noise_type,
                                                   self.noise: noise})
                self.writer_train.add_summary(summary_str, i)

                err_1 = self.loss_1.eval({
                    self.inputs: batch_images,
                    self.labels: batch_labels,
                    self.mode: "TRAIN",
                    self.noise_type:'',
                    self.noise: np.zeros(1000)
                })
                if self.lambda_loss != 0:
                    err_2 = self.loss_2.eval({
                        self.inputs: batch_images,
                        self.labels: batch_labels,
                        self.mode: "TRAIN",
                        self.noise_type:'',
                        self.noise: np.zeros(1000)
                    })
                if np.mod(idx, 10) == 0:

                    if self.lambda_loss!=0:
                        print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f,loss_1: %.8f, loss_2: %.8f" \
                              % (epoch, self.epoch, idx, batch_idxs,
                                 (time.time() - start_time), err_1, err_2))

                    else:
                        print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f,loss_1: %.8f, loss_2: %.8f" \
                              % (epoch, self.epoch, idx, batch_idxs,
                                 (time.time() - start_time), err_1, err_1))

                    summary_str_val = self.sess.run(self.summary,
                                                   feed_dict={
                                                       self.inputs: Xval.reshape(tuple([-1] + self.input_shape)),
                                                       self.labels: yval,
                                                       self.mode: "TEST",
                                                       self.noise_type:'',
                                                       self.noise: np.zeros(1000)})

                    self.writer_val.add_summary(summary_str_val, i)


                i += 1

            pred = np.argmax(self.predict(Xval),axis=1)
            print('Validation accuracy =',np.mean(pred==np.argmax(yval,axis=1)))

            # summary_str_grad_val = self.sess.run(self.loss_2_sum,
            #                                     feed_dict={
            #                                     self.inputs: Xval.reshape(tuple([-1] + self.input_shape)),
            #                                     self.labels: yval,
            #                                     self.mode: "TEST",
            #                                     self.noise_type:'',
            #                                     self.noise: np.zeros(1000)})
            #
            # self.writer_grad_val.add_summary(summary_str_grad_val, i)
            self.save(counter)
            counter += 1

    def predict(self, X, noise_type='',noise=np.zeros(1000)):
        return self.sess.run(self.network, feed_dict={
                                self.inputs: X.reshape(tuple([-1]+self.input_shape)),
                                self.mode: "TEST",
                                self.noise_type:noise_type,
                                self.noise: noise
                            })

    def save(self,step):
        checkpoint_dir = self.checkpoint_dir

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir,self.model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir)  # , self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
