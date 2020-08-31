import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import numpy as np
import os as os
import ns

tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers

def make_dataset(srcdir, eps = 0.00001):
    dataset = None
    for filename in os.listdir(srcdir):
        df = pd.read_csv(os.path.join(srcdir,filename), usecols = ['Thau','Alpha0','Alpha1','Alpha2'])
        lx = df['Thau'].notnull()
        m = df[lx][['Thau','Alpha0','Alpha1','Alpha2']].to_numpy() + eps
        if dataset is None:
            dataset = m
        else:
            dataset = np.concatenate([dataset,m], axis = 0)
    return tf.data.Dataset.from_tensor_slices(dataset).shuffle(1000000)

cross_entropy = tfk.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def custom_reg(weight_matrix):
    return tf.math.reduce_sum(tf.math.sign(weight_matrix))

def make_generator_model(event_shape = 5, alpha = 0.3, wide = 32, deep = 1, use_bias = False, weight_restriction = None, kernel_reg = None):
    inputs = tfk.Input(shape = 2)
    x = tfkl.RepeatVector(event_shape)(inputs)
    x = tfpl.IndependentNormal(dtype='float64')(x)
    for _ in range(deep):
        x = tfkl.Dense(wide, kernel_constraint = weight_restriction, use_bias = use_bias, kernel_regularizer = kernel_reg)(x)
        x = tfkl.LeakyReLU(alpha)(x)
    x = tfkl.Dense(4, kernel_constraint = weight_restriction, use_bias = use_bias, kernel_regularizer=kernel_reg)(x)
    x1 = tfkl.Dense(3,kernel_constraint = weight_restriction, use_bias = use_bias, kernel_regularizer=kernel_reg)(x)
    x1 = tfkl.Lambda(lambda x : tf.math.exp(x))(x1)
    x2 = tfkl.Dense(1,kernel_constraint = weight_restriction, use_bias = use_bias, kernel_regularizer=kernel_reg)(x)
    outputs = tfkl.Concatenate()([x1, x2])
    model = tfk.Model(inputs, outputs)
    return model

@tf.function
def transform_to_yc(x):
    terms = tf.constant([[0.25, 0.5, 1.0, 2.0, 3.0, 4.0,5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 
                                 24.0, 36.0, 48.0, 60.0, 72.0, 84.0, 96.0, 108.0, 120.0, 144.0, 180.0, 240.0, 
                                 300.0, 360.0, 480.0, 600.0]])   
    batch = tf.shape(x)[0]
    sz = tf.constant(terms.shape[1])
    val1 = tf.divide(terms, tf.slice(x, [0,0], [batch,1]))
    val2 = tf.math.exp(-val1)
    val3 = tf.divide(1.0 - val2, val1)
    y = tf.add(tf.concat([tf.ones([batch,sz,1]), tf.zeros([batch,sz,1]), tf.expand_dims(-val2, 2)], axis = -1),
               tf.math.multiply(tf.expand_dims(val3, 2), [[-1.0, 1.0, 1.0]]))
    return  tf.einsum('...ik,...k->...i',y, tf.slice(x, [0 , 1],[batch, 3]))

def make_discriminator_model(wide = 64, deep = 1, alpha = 0.3):
    inputs = tfk.Input(shape = (4,))
    x = tfkl.Lambda(lambda x : transform_to_yc(x))(inputs)
    for _ in range(deep):
        x = tfkl.Dense(wide)(x)
        x = tfkl.LeakyReLU(alpha)(x)
    outputs = tfkl.Dense(1)(x)
    model = tfk.Model(inputs, outputs)
    return model

class GAPNSModel(tfk.Model):
    def __init__(self, event_shape = 5, gwide = 32, gdeep = 1, weight_restriction = None, kernel_reg = None, 
                 use_bias = False, dwide = 32, ddeep = 1, penalty = 0.0001):
        super(GAPNSModel, self).__init__()
        self.generator = make_generator_model(event_shape = event_shape, alpha = 0.3, wide = gwide, deep = gdeep, use_bias = use_bias,
                                              weight_restriction = weight_restriction, kernel_reg = kernel_reg)
        self.discriminator = make_discriminator_model(wide = dwide, deep = ddeep)
        self.generator_optimizer = tfk.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tfk.optimizers.Adam(1e-4)
        self.batch = 16
        self.generator_single_input = tf.constant([[0.0, 1.0]])
        self.generator_inputs = tf.repeat(self.generator_single_input, self.batch, axis = 0)
        self.penalty = tf.constant(penalty)
    
    @tf.function
    def __call__(self, cnt = 1):
        inputs = tf.repeat(self.generator_single_input, cnt, axis = 0)
        return tf.squeeze(self.generator(inputs, training=False))
    
    @tf.function
    def train_step(self, data):        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_nsv = self.generator(self.generator_inputs, training=True)
        
            real_output = self.discriminator(data, training=True)
            fake_output = self.discriminator(generated_nsv, training=True)
            
            penalty = self.penalty * tf.math.reduce_sum(tf.convert_to_tensor(self.generator.losses))
            
            gen_loss = generator_loss(fake_output) - penalty
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables)) 
        
    def fit(self, dataset, epochs = 1, batch = 16):
        self.batch = batch
        self.generator_inputs = tf.repeat(self.generator_single_input, self.batch, axis = 0)
        for epoch in range(epochs):
            print('Epoch - {}'.format(epoch))
            for ns_batch in dataset:
                self.train_step(ns_batch)
                
def generate_ycs(model, max_x = 1200, itercnt = 64, thau_min = 10.0, thau_max = 140.0, max_dev = -0.1):
    x = np.arange(max_x) + 1.0
    x = np.concatenate([np.array([0.25, 0.5, 0.75]), x])
    i2 = x.shape[0] - 1
    nsl = ns.NelsonSiegelLayer()
    for _ in range(itercnt):
        nsp = model()
        if nsp[0] > thau_min and nsp[0] < thau_max:
            nsl.assignValues(nsp)
            y = nsl(x)
            i1 = np.argmax(y)
            if y[i2] - y[i1] > max_dev:
                return (x, y, nsp)
