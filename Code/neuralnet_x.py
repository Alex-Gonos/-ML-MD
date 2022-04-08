import numpy as np
import tensorflow as tf

from tensorflow import keras


np.random.seed(2512517)

class Network(object):
    def __init__(self,dynamical_system,nsteps,dt):
        self.dynamical_system = dynamical_system
        self.dim = self.dynamical_system.dim
        self.dt = dt
        self.nsteps = nsteps
        
    def _build_model(self):
        inputs = keras.Input(shape=(self.nsteps,self.dim))
        q_n = tf.unstack(inputs,axis=1)[-1]

        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        x = keras.layers.Rescaling(self.dt)(x)
        outputs = keras.layers.Add()([q_n,x])
        self.model = keras.Model(inputs=inputs,outputs=outputs)
        self.model.compile(loss='mse',metrics=[],optimizer=keras.optimizers.Adam())
        self.xv = np.zeros((1,self.nsteps,self.dim))        
    
    def set_state(self,x,v):
        '''Set the current state of the integrator
        
        :arg x: Array of size nsteps x dim with initial positions
        :arg v: Array of size nsteps x dim with initial velocities
        '''
        self.xv[0,:,:self.dim] = x[:,:]
#        self.xv[0,:,self.dim//2:] = v[:,:]
        
    @property
    def x(self):
        '''Return the current position vector (as a d-dimensional array)'''
        return self.xv[0,-1,:self.dim]

#    @property
#    def v(self):
#        '''Return the current velocity vector (as a d-dimensional array)'''
#        return self.xv[0,-1,self.dim//2:]
    
    def integrate(self,n_steps):
        '''Carry out a given number of integration steps
        
        :arg n_steps: number of integration steps
        '''
        for k in range(n_steps):
            x_pred = np.asarray(self.model.predict(self.xv)).flatten()
            self.xv = np.roll(self.xv, -1, axis=1)
            self.xv[0,-1,:] = x_pred[:]
            
    # def energy(self):
    #     return self.dynamical_system.energy(self.x,self.v)
    
##Classes for testing dense layer # and width

class NNIntegrator(Network):    
    def __init__(self,dynamical_system,nsteps,dt):
        super().__init__(dynamical_system,nsteps,dt)
        self.dense_layers = [keras.layers.Flatten(),
                    keras.layers.Dense(32,activation='sigmoid'),
                    keras.layers.Dense(64,activation='sigmoid'),
                    keras.layers.Dense(32,activation='sigmoid'), 
                    keras.layers.Dense(self.dim)]
        super()._build_model()
        
 
        
class NN2Integrator(Network):    
    def __init__(self,dynamical_system,nsteps,dt):
        super().__init__(dynamical_system,nsteps,dt)
        self.dense_layers = [keras.layers.Flatten(),
                    keras.layers.Dense(64,activation='sigmoid'), 
                    keras.layers.Dense(self.dim)]
        super()._build_model()



class NN3Integrator(Network):    
    def __init__(self,dynamical_system,nsteps,dt):
        super().__init__(dynamical_system,nsteps,dt)
        self.dense_layers = [keras.layers.Flatten(),
                    keras.layers.Dense(32,activation='sigmoid'),
                    keras.layers.Dense(64,activation='sigmoid'),
                    keras.layers.Dense(self.dim)]
        super()._build_model()        
       
        

class NN4Integrator(Network):    
    def __init__(self,dynamical_system,nsteps,dt):
        super().__init__(dynamical_system,nsteps,dt)
        self.dense_layers = [keras.layers.Flatten(),
                    keras.layers.Dense(64,activation='sigmoid'),
                    keras.layers.Dense(64,activation='sigmoid'),
                    keras.layers.Dense(self.dim)]
        super()._build_model()       



class NN5Integrator(Network):    
    def __init__(self,dynamical_system,nsteps,dt):
        super().__init__(dynamical_system,nsteps,dt)
        self.dense_layers = [keras.layers.Flatten(),
                    keras.layers.Dense(64,activation='sigmoid'),
                    keras.layers.Dense(64,activation='sigmoid'),
                    keras.layers.Dense(32,activation='sigmoid'),
                    keras.layers.Dense(self.dim)]
        super()._build_model()   
        
        
        
class NN6Integrator(Network):    
    def __init__(self,dynamical_system,nsteps,dt):
        super().__init__(dynamical_system,nsteps,dt)
        self.dense_layers = [keras.layers.Flatten(),
                    keras.layers.Dense(64,activation='sigmoid'),
                    keras.layers.Dense(64,activation='sigmoid'),
                    keras.layers.Dense(64,activation='sigmoid'),
                    keras.layers.Dense(self.dim)]
        super()._build_model()   
        
        
##Classes for testing different learning rates

class NN5LR2Integrator(Network):    
    def __init__(self,dynamical_system,nsteps,dt):
        super().__init__(dynamical_system,nsteps,dt)
        self.dense_layers = [keras.layers.Flatten(),
                    keras.layers.Dense(64,activation='relu'),
                    keras.layers.Dense(64,activation='relu'),
                    keras.layers.Dense(32,activation='relu'),
                    keras.layers.Dense(self.dim)]
        self._build_model()
        
    def _build_model(self):
        inputs = keras.Input(shape=(self.nsteps,self.dim))
        q_n = tf.unstack(inputs,axis=1)[-1]

        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        x = keras.layers.Rescaling(self.dt)(x)
        outputs = keras.layers.Add()([q_n,x])
        self.model = keras.Model(inputs=inputs,outputs=outputs)
        self.model.compile(loss='mse',metrics=[],optimizer=keras.optimizers.Adam(learning_rate=0.0005))
        self.xv = np.zeros((1,self.nsteps,self.dim)) 



class NN5LR3Integrator(Network):    
    def __init__(self,dynamical_system,nsteps,dt):
        super().__init__(dynamical_system,nsteps,dt)
        self.dense_layers = [keras.layers.Flatten(),
                    keras.layers.Dense(64,activation='relu'),
                    keras.layers.Dense(64,activation='relu'),
                    keras.layers.Dense(32,activation='relu'),
                    keras.layers.Dense(self.dim)]
        self._build_model()
        
    def _build_model(self):
        inputs = keras.Input(shape=(self.nsteps,self.dim))
        q_n = tf.unstack(inputs,axis=1)[-1]

        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        x = keras.layers.Rescaling(self.dt)(x)
        outputs = keras.layers.Add()([q_n,x])
        self.model = keras.Model(inputs=inputs,outputs=outputs)
        self.model.compile(loss='mse',metrics=[],optimizer=keras.optimizers.Adam(learning_rate=0.0001))
        self.xv = np.zeros((1,self.nsteps,self.dim)) 


#NN5LR3 is our final choice of DNN integrator


##Class for testing approximating state directly
        
class NNStateIntegrator(Network):    
    def __init__(self,dynamical_system,nsteps,dt):
        super().__init__(dynamical_system,nsteps,dt)
        self.dense_layers = [keras.layers.Flatten(),
                    keras.layers.Dense(32,activation='sigmoid'),
                    keras.layers.Dense(64,activation='sigmoid'),
                    keras.layers.Dense(32,activation='sigmoid'), 
                    keras.layers.Dense(self.dim)]
        self._build_model()
        
    def _build_model(self):
        inputs = keras.Input(shape=(self.nsteps,self.dim))
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        outputs = x
        self.model = keras.Model(inputs=inputs,outputs=outputs)
        self.model.compile(loss='mse',metrics=[],optimizer=keras.optimizers.Adam())
        self.xv = np.zeros((1,self.nsteps,self.dim))    
        
##Class for testing normalisation methods        
        
class NNNormIntegrator(Network):    
    def __init__(self,dynamical_system,nsteps,dt):
        super().__init__(dynamical_system,nsteps,dt)
        self.dense_layers = [keras.layers.Flatten(),
                    keras.layers.Dense(64),
                    keras.layers.BatchNormalization(),
                    keras.layers.Activation('relu'),
                    keras.layers.Dropout(rate=0.15),
                    keras.layers.Dense(64),
                    keras.layers.BatchNormalization(),
                    keras.layers.Activation('relu'),
                    keras.layers.Dropout(rate=0.15),
                    keras.layers.Dense(32,activation='relu'), 
                    keras.layers.Dense(self.dim)]
        self._build_model()
        
    def _build_model(self):
        inputs = keras.Input(shape=(self.nsteps,self.dim))
        q_n = tf.unstack(inputs,axis=1)[-1]

        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        x = keras.layers.Rescaling(self.dt)(x)
        outputs = keras.layers.Add()([q_n,x])
        self.model = keras.Model(inputs=inputs,outputs=outputs)
        self.model.compile(loss='mse',metrics=[],optimizer=keras.optimizers.Adam(learning_rate=0.0001))
        self.xv = np.zeros((1,self.nsteps,self.dim))     

##Classes for testing different activation functions

class NNReluIntegrator(Network):    
    def __init__(self,dynamical_system,nsteps,dt):
        super().__init__(dynamical_system,nsteps,dt)
        self.dense_layers = [keras.layers.Flatten(),
                    keras.layers.Dense(64,activation='relu'),
                    keras.layers.Dense(64,activation='relu'),
                    keras.layers.Dense(32,activation='relu'), 
                    keras.layers.Dense(self.dim)]
        super()._build_model()



class NNTanhIntegrator(Network):    
    def __init__(self,dynamical_system,nsteps,dt):
        super().__init__(dynamical_system,nsteps,dt)
        self.dense_layers = [keras.layers.Flatten(),
                    keras.layers.Dense(64,activation='tanh'),
                    keras.layers.Dense(64,activation='tanh'),
                    keras.layers.Dense(32,activation='tanh'), 
                    keras.layers.Dense(self.dim)]
        super()._build_model()




##LSTM Classes


#LSTM Classes for architecture comparison


class LSTMIntegrator(Network): 
    def __init__(self,dynamical_system,nsteps,dt):
        super().__init__(dynamical_system,nsteps,dt)
        self.dense_layers = [keras.layers.LSTM(32,return_sequences=True),
                        keras.layers.LSTM(32),
                        keras.layers.Dense(self.dim)]
        super()._build_model()
        


class LSTM2Integrator(Network): 
    def __init__(self,dynamical_system,nsteps,dt):
        super().__init__(dynamical_system,nsteps,dt)
        self.dense_layers = [keras.layers.LSTM(64,return_sequences=True),
                        keras.layers.LSTM(64),
                        keras.layers.Dense(self.dim)]
        super()._build_model()



class LSTM3Integrator(Network): 
    def __init__(self,dynamical_system,nsteps,dt):
        super().__init__(dynamical_system,nsteps,dt)
        self.dense_layers = [keras.layers.LSTM(32,return_sequences=True),
                        keras.layers.LSTM(32),
                        keras.layers.Dense(64,activation='sigmoid'),
                        keras.layers.Dense(self.dim)]
        super()._build_model()



class LSTM4Integrator(Network): 
    def __init__(self,dynamical_system,nsteps,dt):
        super().__init__(dynamical_system,nsteps,dt)
        self.dense_layers = [keras.layers.LSTM(64,return_sequences=True),
                        keras.layers.LSTM(64),
                        keras.layers.Dense(32,activation='sigmoid'),
                        keras.layers.Dense(self.dim)]
        super()._build_model()



class LSTM5Integrator(Network): 
    def __init__(self,dynamical_system,nsteps,dt):
        super().__init__(dynamical_system,nsteps,dt)
        self.dense_layers = [keras.layers.LSTM(64,return_sequences=True),
                        keras.layers.LSTM(64),
                        keras.layers.Dense(64,activation='sigmoid'),
                        keras.layers.Dense(self.dim)]
        super()._build_model()



class LSTM6Integrator(Network): 
    def __init__(self,dynamical_system,nsteps,dt):
        super().__init__(dynamical_system,nsteps,dt)
        self.dense_layers = [keras.layers.LSTM(64,return_sequences=True),
                        keras.layers.LSTM(64,return_sequences=True),
                        keras.layers.LSTM(64),
                        keras.layers.Dense(self.dim)]
        super()._build_model()
      
        
        
class LSTM7Integrator(Network): 
    def __init__(self,dynamical_system,nsteps,dt):
        super().__init__(dynamical_system,nsteps,dt)
        self.dense_layers = [keras.layers.LSTM(64,return_sequences=True),
                             keras.layers.LSTM(64,return_sequences=True),
                        keras.layers.LSTM(32),
                        keras.layers.Dense(self.dim)]
        self._build_model()  



class LSTM8Integrator(Network): 
    def __init__(self,dynamical_system,nsteps,dt):
        super().__init__(dynamical_system,nsteps,dt)
        self.dense_layers = [keras.layers.LSTM(32,return_sequences=True),
                             keras.layers.LSTM(32,return_sequences=True),
                        keras.layers.LSTM(32),
                        keras.layers.Dense(self.dim)]
        self._build_model()  
        
        

class LSTM9Integrator(Network): 
    def __init__(self,dynamical_system,nsteps,dt):
        super().__init__(dynamical_system,nsteps,dt)
        self.dense_layers = [keras.layers.LSTM(32,return_sequences=True),
                             keras.layers.LSTM(32,return_sequences=True),
                        keras.layers.Dense(32,activation='sigmoid'),
                        keras.layers.Dense(self.dim)]
        self._build_model()  
        

#LSTM Activation Function Comparison

class LSTMReluIntegrator(Network): 
    def __init__(self,dynamical_system,nsteps,dt):
        super().__init__(dynamical_system,nsteps,dt)
        self.dense_layers = [keras.layers.LSTM(64,return_sequences=True),
                        keras.layers.LSTM(64),
                        keras.layers.Dense(64,activation='relu'),
                        keras.layers.Dense(self.dim)]
        super()._build_model()


class LSTMRelu2Integrator(Network): 
    def __init__(self,dynamical_system,nsteps,dt):
        super().__init__(dynamical_system,nsteps,dt)
        self.dense_layers = [keras.layers.LSTM(64,return_sequences=True),
                        keras.layers.Activation('relu'),
                        keras.layers.LSTM(64),
                        keras.layers.Activation('relu'),
                        keras.layers.Dense(64,activation='relu'),
                        keras.layers.Dense(self.dim)]
        super()._build_model()
        
        

class LSTMTanhIntegrator(Network): 
    def __init__(self,dynamical_system,nsteps,dt):
        super().__init__(dynamical_system,nsteps,dt)
        self.dense_layers = [keras.layers.LSTM(64,return_sequences=True),
                        keras.layers.LSTM(64),
                        keras.layers.Dense(64,activation='tanh'),
                        keras.layers.Dense(self.dim)]
        super()._build_model()
                
     
 
class LSTMTanh2Integrator(Network): 
    def __init__(self,dynamical_system,nsteps,dt):
        super().__init__(dynamical_system,nsteps,dt)
        self.dense_layers = [keras.layers.LSTM(64,return_sequences=True),
                        keras.layers.Activation('tanh'),     
                        keras.layers.LSTM(64),
                        keras.layers.Activation('tanh'),
                        keras.layers.Dense(64,activation='tanh'),
                        keras.layers.Dense(self.dim)]
        super()._build_model()       
     
        
#Class for testing normalisation methods on LSTM        
        
class LSTMNormIntegrator(Network): 
    def __init__(self,dynamical_system,nsteps,dt):
        super().__init__(dynamical_system,nsteps,dt)
        self.dense_layers = [keras.layers.LSTM(64,return_sequences=True),
                             keras.layers.BatchNormalization(),
                             keras.layers.Activation('relu'),
                             keras.layers.Dropout(rate=0.15),
                        keras.layers.LSTM(64),
                        keras.layers.BatchNormalization(),
                        keras.layers.Activation('relu'),
                        keras.layers.Dropout(rate=0.15),
                        keras.layers.Dense(32,activation='tanh'),
                        keras.layers.Dense(self.dim)]
        self._build_model()     
        
    def _build_model(self):
        inputs = keras.Input(shape=(self.nsteps,self.dim))
        q_n = tf.unstack(inputs,axis=1)[-1]

        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        x = keras.layers.Rescaling(self.dt)(x)
        outputs = keras.layers.Add()([q_n,x])
        self.model = keras.Model(inputs=inputs,outputs=outputs)
        self.model.compile(loss='mse',metrics=[],optimizer=keras.optimizers.Adam(learning_rate=0.0001))
        self.xv = np.zeros((1,self.nsteps,self.dim)) 
        
###class for testing approximating state directly

class LSTMStateIntegrator(Network):    
    def __init__(self,dynamical_system,nsteps,dt):
        super().__init__(dynamical_system,nsteps,dt)
        self.dense_layers = [keras.layers.LSTM(64,return_sequences=True),
                        keras.layers.LSTM(64),
                        keras.layers.Dense(64,activation='tanh'),
                        keras.layers.Dense(self.dim)]
        self._build_model()
        
    def _build_model(self):
        inputs = keras.Input(shape=(self.nsteps,self.dim))
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        outputs = x
        self.model = keras.Model(inputs=inputs,outputs=outputs)
        self.model.compile(loss='mse',metrics=[],optimizer=keras.optimizers.Adam(learning_rate=0.0001))
        self.xv = np.zeros((1,self.nsteps,self.dim))    
        
        
        
class LSTMFinalIntegrator(Network): 
    def __init__(self,dynamical_system,nsteps,dt):
        super().__init__(dynamical_system,nsteps,dt)
        self.dense_layers = [keras.layers.LSTM(64,return_sequences=True),
                        keras.layers.LSTM(64),
                        keras.layers.Dense(64,activation='tanh'),
                        keras.layers.Dense(self.dim)]
        self._build_model()     
        
    def _build_model(self):
        inputs = keras.Input(shape=(self.nsteps,self.dim))
        q_n = tf.unstack(inputs,axis=1)[-1]

        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        x = keras.layers.Rescaling(self.dt)(x)
        outputs = keras.layers.Add()([q_n,x])
        self.model = keras.Model(inputs=inputs,outputs=outputs)
        self.model.compile(loss='mse',metrics=[],optimizer=keras.optimizers.Adam(learning_rate=0.0001))
        self.xv = np.zeros((1,self.nsteps,self.dim)) 


class LSTMFinal2Integrator(Network): 
    def __init__(self,dynamical_system,nsteps,dt):
        super().__init__(dynamical_system,nsteps,dt)
        self.dense_layers = [keras.layers.LSTM(64,return_sequences=True),
                        keras.layers.LSTM(64,return_sequences=True),
                        keras.layers.LSTM(64),
                        keras.layers.Dense(self.dim)]
        self._build_model()     
        
    def _build_model(self):
        inputs = keras.Input(shape=(self.nsteps,self.dim))
        q_n = tf.unstack(inputs,axis=1)[-1]

        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        x = keras.layers.Rescaling(self.dt)(x)
        outputs = keras.layers.Add()([q_n,x])
        self.model = keras.Model(inputs=inputs,outputs=outputs)
        self.model.compile(loss='mse',metrics=[],optimizer=keras.optimizers.Adam(learning_rate=0.0001))
        self.xv = np.zeros((1,self.nsteps,self.dim)) 

##Class mimicking the design of the MD paper RNN

class LSTMPaperIntegrator(Network): 
    def __init__(self,dynamical_system,nsteps,dt):
        super().__init__(dynamical_system,nsteps,dt)
        self.dense_layers = [keras.layers.LSTM(32,return_sequences=True),
                             keras.layers.BatchNormalization(),
                             keras.layers.Activation('relu'),
                             keras.layers.Dropout(rate=0.15),
                        keras.layers.LSTM(32),
                        keras.layers.BatchNormalization(),
                        keras.layers.Activation('relu'),
                        keras.layers.Dropout(rate=0.15),
                        keras.layers.Dense(self.dim)]
        self._build_model()     
        
    def _build_model(self):
        inputs = keras.Input(shape=(self.nsteps,self.dim))
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        outputs = x
        self.model = keras.Model(inputs=inputs,outputs=outputs)
        self.model.compile(loss='mse',metrics=[],optimizer=keras.optimizers.Adam(learning_rate=0.0005))
        self.xv = np.zeros((1,self.nsteps,self.dim))  
 
        
 
    
 
    
 
    


class BRNNIntegrator(Network):
    def __init__(self,dynamical_system,nsteps,dt):
        super().__init__(dynamical_system,nsteps,dt)
        self.dense_layers = [keras.layers.Bidirectional(keras.layers.LSTM(64,return_sequences=True), merge_mode="concat"),
                        keras.layers.Bidirectional(keras.layers.LSTM(64,return_sequences=True), merge_mode="concat"),
                        keras.layers.Bidirectional(keras.layers.LSTM(64), merge_mode="ave"),
                        keras.layers.Dense(32,activation='tanh'),
                        keras.layers.Dense(self.dim)]
        super()._build_model()
    
    
    
    
    
    
    
    
    
class DataGenerator(object):
    def __init__(self,nn_integrator,train_integrator):        
        self.nn_integrator = nn_integrator
        self.train_integrator = train_integrator
        self.dynamical_system = self.nn_integrator.dynamical_system
        self.dataset = tf.data.Dataset.from_generator(self._generator,                                                      
                                                      output_signature=(
                                                          tf.TensorSpec(shape=(self.nn_integrator.nsteps,
                                                                               self.dynamical_system.dim), dtype=tf.float32),
                                                          tf.TensorSpec(shape=(self.dynamical_system.dim), dtype=tf.float32)
                                                      ))
    
    def _generator(self):
        state = np.zeros((self.nn_integrator.nsteps+1,2*self.dynamical_system.dim))
        while True:
            self.dynamical_system.set_random_state(state[0,:self.dynamical_system.dim],
                                                   state[0,self.dynamical_system.dim:])
            self.train_integrator.set_state(state[0,:self.dynamical_system.dim],
                                            state[0,self.dynamical_system.dim:])
            for k in range(self.nn_integrator.nsteps):
                self.train_integrator.integrate(int(self.nn_integrator.dt/self.train_integrator.dt))
                state[k+1,:self.dynamical_system.dim] = self.train_integrator.x[:]
                state[k+1,self.dynamical_system.dim:] = self.train_integrator.v[:]
            out = state[:,0:self.dynamical_system.dim]
            X = out[:-1,:]
            y = out[-1,:]
            yield (X,y)    
    