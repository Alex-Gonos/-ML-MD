import numpy as np
import tensorflow as tf
from tensorflow import keras


np.random.seed(2512517)

class Network(object):
    def __init__(self,dynamical_system,nsteps,dt):
        self.dynamical_system = dynamical_system
        self.dim = 2*self.dynamical_system.dim
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
        self.xv[0,:,:self.dim//2] = x[:,:]
        self.xv[0,:,self.dim//2:] = v[:,:]
        
    @property
    def x(self):
        '''Return the current position vector (as a d-dimensional array)'''
        return self.xv[0,-1,:self.dim//2]

    @property
    def v(self):
        '''Return the current velocity vector (as a d-dimensional array)'''
        return self.xv[0,-1,self.dim//2:]
    
    def integrate(self,n_steps):
        '''Carry out a given number of integration steps
        
        :arg n_steps: number of integration steps
        '''
        for k in range(n_steps):
            x_pred = np.asarray(self.model.predict(self.xv)).flatten()
            self.xv = np.roll(self.xv, -1, axis=1)
            self.xv[0,-1,:] = x_pred[:]
            
    def energy(self):
        return self.dynamical_system.energy(self.x,self.v)
    
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
        'To use cosine learning rate schedule, uncomment this snippet and replace learning rate with =lr_schedule'
        'To use fixed LR set to 0.0001'
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-4,
            decay_steps=1000*1000,
            alpha=1.E-3)
        self.model.compile(loss='mse',metrics=[],optimizer=keras.optimizers.Adam(learning_rate=lr_schedule))
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
        self.dense_layers = [keras.layers.LSTM(32,return_sequences=True),
                        keras.layers.LSTM(32),
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
        'To use cosine learning rate schedule, uncomment this snippet and replace learning rate with =lr_schedule'
        'To use fixed LR set to 0.0001'
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-4,
            decay_steps=1000*1000,
            alpha=1.E-3)
        self.model.compile(loss='mse',metrics=[],optimizer=keras.optimizers.Adam(learning_rate=lr_schedule))
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




    
    
###Classes necessary for hamiltonian  NN integrator    
    
class HamiltonianNNIntegrator(NNIntegrator):
    '''Neural network integrator based on the Hamiltonian Stoermer-Verlet update'''
    def __init__(self,dynamical_system,dt,V_pot_layers,T_kin_layers):
        super().__init__(dynamical_system,1,dt)
        self.V_pot_layers = V_pot_layers
        self.T_kin_layers = T_kin_layers
        self._build_model()
    
    def _build_model(self):
        self.model = VerletModel(self.dim,self.dt,
                                 self.V_pot_layers,
                                 self.T_kin_layers)
        'To use cosine learning rate schedule, uncomment this snippet and replace learning rate with =lr_schedule'
        'To use fixed LR set to 0.0001'
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-4,
            decay_steps=1000*1000,
            alpha=1.E-3)
        self.model.build(input_shape=(None,1,self.dim))
        self.model.compile(loss='mse',metrics=[],optimizer=keras.optimizers.Adam(learning_rate=lr_schedule))    
    
    
    
    
    
    
    
class DataGenerator(object):
    def __init__(self,nn_integrator,train_integrator):        
        self.nn_integrator = nn_integrator
        self.train_integrator = train_integrator
        self.dynamical_system = self.nn_integrator.dynamical_system
        self.dataset = tf.data.Dataset.from_generator(self._generator,                                                      
                                                      output_signature=(
                                                          tf.TensorSpec(shape=(self.nn_integrator.nsteps,
                                                                               2*self.dynamical_system.dim), dtype=tf.float32),
                                                          tf.TensorSpec(shape=(2*self.dynamical_system.dim), dtype=tf.float32)
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
            X = state[:-1,:]
            y = state[-1,:]
            yield (X,y)    
    
    
    
    
    
# Altered Verlet Model for use with Hamiltonian NN

class VerletModel(keras.Model):
    '''Single step of a Symplectic Stoermer Verlet integrator update for a 
    separable system with Hamiltonian H(q,p) = T(p) + V(q)
    
    The model maps the current state (q_n,p_n) to next state (q_{n+1},p_{n+1}) 
    using the update
    
    p_{n+1/2} = p_n - dt/2*dV/dq(q_n)
    q_{n+1} = q_n + dt*dT/dp(p_{n+1/2})
    p_{n+1} = p_{n+1/2} - dt/2*dV/dq(q_{n+1})
    
    Both the kinetic energy T(p) and the potential energy V(q) are represented 
    by neural networks. The position q_n and momentum p_n are d-dimensional vectors.
    
    :arg dim: dimension d of the Hamiltonian system 
    :arg dt: timestep size
    :arg V_pot_layers: layers encoding the neural network for potential energy V(q)
    :arg T_kin_layers: layers encoding the neural network for kinetic energy T(p)
    '''
    
    def __init__(self,dim,dt,V_pot_layers,T_kin_layers):
        super().__init__()
        self.dim = dim
        self.dt = dt
        self.V_pot_layers = V_pot_layers
        self.T_kin_layers = T_kin_layers
        self.V_pot_final_layer = keras.layers.Dense(self.dim//2,use_bias=False)
        self.T_kin_final_layer = keras.layers.Dense(self.dim//2,use_bias=False)
    
    #@tf.function
    def V_pot(self,q):
        '''Evaluate potential energy network V(q)
        
        :arg q: position q  which to evaluate the potential
        '''
        x = q
        for layer in self.V_pot_layers:
            x = layer(x)
        x = self.V_pot_final_layer(x)
        return x
        
        
    #@tf.function
    def T_kin(self,p):
        '''Evaluate kinetic energy network T(p)
        
        :arg p: momentum p at which to evaluate the kinetic energy
        '''
        x = p
        for layer in self.T_kin_layers:
            x = layer(x)
        x = self.T_kin_final_layer(x)
        return x
        

    @tf.function
    def verlet_step(self,q_n,p_n):
        '''Carry out a single Stoermer-Verlet step
        
        This function maps (q_n,p_n) to (q_{n+1},p_{n+1}) using a single Stoermer
        Verlet step
        
        :arg q_n: current position q_n
        :arg p_n: current momentum p_n
        '''
        # p_{n+1/2} = p_n - dt/2*dV/dq(q_n)
        dV_dq = tf.gradients(self.V_pot(q_n),q_n)[0]
        p_n = p_n - 0.5*self.dt*dV_dq

        # q_{n+1} = q_n + dt*dT/dq(p_{n+1/2})
        dT_dp = tf.gradients(self.T_kin(p_n),p_n)[0]
        q_n = q_n + self.dt*dT_dp

        # p_{n+1} = p_{n+1/2} - dt/2*dV/dq(q_{n+1})
        dV_dq = tf.gradients(self.V_pot(q_n),q_n)[0]
        p_n = p_n - 0.5*self.dt*dV_dq
        
        return q_n, p_n

    def call(self, inputs):
        '''Evaluate model
        
        Split the inputs = (q_n,p_n) into position and momentum and 
        return the state (q_{n+1},p_{n+1}) at the next timestep.
        
        Note that the expected tensor shape is B x 1 x 2d to be compatible with
        the non-symplectic update 
        
        :arg inputs: state (q_n,p_n) as a B x 1 x 2d tensor
        '''
        
        input_shape = tf.shape(inputs)
        # Extract q_n and p_n from input
        qp_old = tf.unstack(tf.reshape(inputs, (input_shape[0],input_shape[2],)),axis=-1)
        q_old = tf.stack(qp_old[:self.dim//2],axis=-1)
        p_old = tf.stack(qp_old[self.dim//2:],axis=-1)
        q_new, p_new = self.verlet_step(q_old,p_old)        
        # Combine result of Verlet step into tensor of correct size
        outputs = tf.concat([q_new,p_new],axis=-1)
        return outputs    
    