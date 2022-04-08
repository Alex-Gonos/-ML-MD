import numpy as np
import math
import ctypes
import subprocess
import string

class DynamicalSystem(object):
    def __init__(self,dim,mass):
        '''Abstract base class for a dynamical system

        Models a d-dimensional system of the following form:

          dx_j/dt = v_j
          dv_j/dt = F_j(x)/m_j

        where j = 0,1,2,...,d-1

        :arg dim: Spatial dimension of dynamical system
        :arg mass: Mass of the system (can be a scalar in 1d)
        '''
        self.dim = dim
        self.mass = mass

    def compute_scaled_force(self,x,v,force):
        '''Store the forces scaled by inverse mass in the vector
        such that force[j] = F_j(x)/m_j

        :arg x: Particle positions x (d-dimensional array)
        :arg force: Resulting force vector (d-dimensional array)
        '''
        pass

    def set_random_state(self,x,v):
        '''Set the position x and v to random values. This will be used
        during the training stage to pick a suitable set of initial values
        for the problem at hand

        :arg x: Positions (d-dimensional array)
        :arg v: Velocities (d-dimensional array)
        '''
        pass


    def energy(self,x,v):
        '''Return the total energy for given positions and velocities

        :arg x: Positions (d-dimensional array)
        :arg v: Velocities (d-dimensional array)
        '''
        pass
    
    
    def forward_map(self,x0,v0,t):
        '''Exact forward map
        
        Compute position x(t) and velocity v(t), given initial position x(0) and velocity v(0).
        This will only be implemented if the specific dynamical system has an analytical solution
        
        :arg x0: initial position x(0)
        :arg v0: initial velocity v(0)
        :arg t: final time
        '''
        raise NotImplementedError("Dynamical system has no exact solution.")
    

class HarmonicOscillator(DynamicalSystem):
    def __init__(self,mass,k_spring):
        '''One-dimensional harmonic oscillator described by the equations
        of motion

        dx_0/dt = v_0, dv_0/dt = -k/m_0*x_0

        :arg mass: Particle mass
        :arg k_spring: Spring constant k
        '''
        super().__init__(1,mass)
        self.k_spring = k_spring

    def compute_scaled_force(self,x,v,force):
        '''Set the entry force[0] of the force vector
        to -k/m_0*x_0

        :arg x: Particle position x (1-dimensional array)
        :arg force: Resulting force vector (1-dimensional array)
        '''
        force[0] = -self.k_spring/self.mass*x[0]

    def set_random_state(self,x,v):
        '''Draw position and velocity from a normal distribution
        :arg x: Positions (d-dimensional array)
        :arg v: Velocities (d-dimensional array)
        arg mass: Mass
        arg k_spring: Spring constant k
        '''
        x[0] = np.random.normal(0,3)
        v[0] = np.random.normal(0,1)
        #self.mass = np.random.randint(1,10)
        #self.k_spring = np.random.randint(1,10)

    def energy(self,x,v):
        '''Compute total energy E = 1/2*m*v_0^2 + 1/2*k*x_0^2

        :arg x: Positions (d-dimensional array)
        :arg v: Velocities (d-dimensional array)
        '''
        return 0.5*self.mass*v[0]**2 + 0.5*self.k_spring*x[0]**2
    
    def forward_map(self,x0,v0,t):
        '''Exact forward map
        
        Compute position x(t) and velocity v(t), given initial position x(0) and velocity v(0).
        
        For this use:
        
        x(t) = x(0)*cos(omega*t) + omega*v(0)*sin(omega*t)
        v(t) = -x(0)/omega*sin(omega*t) + v(0)*cos(omega*t)
        
        with omegae = sqrt(k/m)
        
        :arg x0: initial position x(0)
        :arg v0: initial velocity v(0)
        :arg t: final time
        '''
        omega = np.sqrt(self.k_spring/self.mass)
        cos_omegat = np.cos(omega*t)
        sin_omegat = np.sin(omega*t)
        x = np.array(x0[0]*cos_omegat + v0[0]/omega*sin_omegat)
        v = np.array(-x0[0]*omega*sin_omegat + v0[0]*cos_omegat)
        return x, v    

class HarmonicOscillator2(DynamicalSystem):
    def __init__(self,mass,k_spring):
        '''One-dimensional harmonic oscillator described by the equations
        of motion

        dx_0/dt = v_0, dv_0/dt = -k/m_0*x_0

        :arg mass: Particle mass
        :arg k_spring: Spring constant k
        '''
        super().__init__(1,mass)
        self.k_spring = k_spring

    def compute_scaled_force(self,x,v,force):
        '''Set the entry force[0] of the force vector
        to -k/m_0*x_0

        :arg x: Particle position x (1-dimensional array)
        :arg force: Resulting force vector (1-dimensional array)
        '''
        force[0] = -self.k_spring/self.mass*x[0]

    def set_random_state(self,x,v):
        '''Draw position and velocity from a normal distribution
        :arg x: Positions (d-dimensional array)
        :arg v: Velocities (d-dimensional array)
        arg mass: Mass
        arg k_spring: Spring constant k
        '''
        x[0] = np.random.normal(0,3)
        v[0] = np.random.normal(0,3)
        #self.mass = np.random.randint(1,10)
        #self.k_spring = np.random.randint(1,10)

    def energy(self,x,v):
        '''Compute total energy E = 1/2*m*v_0^2 + 1/2*k*x_0^2

        :arg x: Positions (d-dimensional array)
        :arg v: Velocities (d-dimensional array)
        '''
        return 0.5*self.mass*v[0]**2 + 0.5*self.k_spring*x[0]**2
    
    def forward_map(self,x0,v0,t):
        '''Exact forward map
        
        Compute position x(t) and velocity v(t), given initial position x(0) and velocity v(0).
        
        For this use:
        
        x(t) = x(0)*cos(omega*t) + omega*v(0)*sin(omega*t)
        v(t) = -x(0)/omega*sin(omega*t) + v(0)*cos(omega*t)
        
        with omegae = sqrt(k/m)
        
        :arg x0: initial position x(0)
        :arg v0: initial velocity v(0)
        :arg t: final time
        '''
        omega = np.sqrt(self.k_spring/self.mass)
        cos_omegat = np.cos(omega*t)
        sin_omegat = np.sin(omega*t)
        x = np.array(x0[0]*cos_omegat + v0[0]/omega*sin_omegat)
        v = np.array(-x0[0]*omega*sin_omegat + v0[0]*cos_omegat)
        return x, v
    
    
class LennardJonesOscillator(DynamicalSystem):
    def __init__(self,mass):
        '''One-dimensional harmonic oscillator described by the equations
        of motion

        dx_0/dt = v_0, dv_0/dt = -(-12*q^-13 + 12*q^-7)

        :arg mass: Particle mass
        '''
        
        super().__init__(1,mass)

    def compute_scaled_force(self,x,v,force):
        '''Set the entry force[0] of the force vector
        to -k/m_0*x_0

        :arg x: Particle position x (1-dimensional array)
        :arg force: Resulting force vector (1-dimensional array)
        '''
        force[0] = -(-12*x[0]**-13 + 12*x[0]**-7)/self.mass

    def set_random_state(self,x,v):
        '''Draw position and velocity from a normal distribution
        :arg x: Positions (d-dimensional array)
        :arg v: Velocities (d-dimensional array)
        arg mass: Mass
        '''
        x[0] = np.random.normal(2,1)
        v[0] = np.random.normal(0,1)
        #self.mass = np.random.randint(1,10)

    def energy(self,x,v):
        '''Compute total energy E = 1/2*v_0^2 + 1/2*k*x_0^2

        :arg x: Positions (d-dimensional array)
        :arg v: Velocities (d-dimensional array)
        '''
        return 0.5*self.mass*v[0]**2 + x[0]**(-12) - x[0]**(-6)
    
class DoubleWell(DynamicalSystem):
    def __init__(self,mass):
        '''One-dimensional Double well described by the equations
        of motion

        dx_0/dt = v_0, dv_0/dt = x-x^3

        :arg mass: Particle mass
        '''
        
        super().__init__(1,mass)

    def compute_scaled_force(self,x,v,force):
        '''Set the entry force[0] of the force vector
        to x-x^3

        :arg x: Particle position x (1-dimensional array)
        :arg force: Resulting force vector (1-dimensional array)
        '''
        force[0] = (x[0] - x[0]**3)/self.mass

    def set_random_state(self,x,v):
        '''Draw position and velocity from a normal distribution
        :arg x: Positions (d-dimensional array)
        :arg v: Velocities (d-dimensional array)
        arg mass: Mass
        '''
        x[0] = np.random.normal(0,1.5)
        v[0] = np.random.normal(0,1)
        #self.mass = np.random.randint(1,10)

    def energy(self,x,v):
        '''Compute total energy E = 1/2*v_0^2 + 1/4x_0^4 + 1/2x_0^2

        :arg x: Positions (d-dimensional array)
        :arg v: Velocities (d-dimensional array)
        '''
        return 0.5*self.mass*v[0]**2 + 0.25*x[0]**4 - 0.5*x[0]**2
    
class Rugged(DynamicalSystem):
    def __init__(self,mass):
        '''One-dimensional Rugged potential described by the equations
        of motion

        dx_0/dt = v_0, dv_0/dt = 1/50(-4x_0^3+3x_0^2+32x_0-4)-6cos(30(x_0+5))

        :arg mass: Particle mass
        '''
        
        super().__init__(1,mass)

    def compute_scaled_force(self,x,v,force):
        '''Set the entry force[0] of the force vector
        to x-x^3

        :arg x: Particle position x (1-dimensional array)
        :arg force: Resulting force vector (1-dimensional array)
        '''
        force[0] = ((1/50) * (-4*x[0]**3 + 3*x[0]**2 + 32*x[0] -4) - 6*math.cos(30*(x[0]+5)))/self.mass

    def set_random_state(self,x,v):
        '''Draw position and velocity from a normal distribution
        :arg x: Positions (d-dimensional array)
        :arg v: Velocities (d-dimensional array)
        arg mass: Mass
        '''
        x[0] = np.random.normal(0,2.5)
        v[0] = np.random.normal(0,1)
        #self.mass = np.random.randint(1,10)

    def energy(self,x,v):
        '''Compute total energy E = 1/2*v_0^2 + 1/4x_0^4 + 1/2x_0^2

        :arg x: Positions (d-dimensional array)
        :arg v: Velocities (d-dimensional array)
        '''
        return 0.5*self.mass*v[0]**2 + (1/50) * (x[0]**4 - x[0]**3 - 16*x[0]**2 +4*x[0] +48) + (1/5) * math.sin(30*(x[0]+5))

class ThreeBody(DynamicalSystem):
    def __init__(self, mass, G):
        '''3-dimensional three body problem described by the equations
        of motion

        dx_0/dt = v_0, dv_0/dt = [force]

        :arg mass: Particle mass (3-vector)
        :arg G: Universal Gravitational Constant (6.674x10^-11)
        '''
        
        super().__init__(9,mass)
        self.G = G


    def compute_scaled_force(self, x, v, force):
        '''Set the entry force[0] of the force vector

        :arg x: cartesian coordinates of the three bodies (9-dimensional array)
        :arg force: Resulting force vector (9-dimensional array)
        '''

        mass = self.mass
               
        force[0:3] = -self.G*(mass[1]*(x[0:3]-x[3:6])/(np.linalg.norm(x[0:3]-x[3:6])**3) + mass[2]*(x[0:3]-x[6:9])/(np.linalg.norm(x[0:3]-x[6:9])**3))
        force[3:6] = -self.G*(mass[2]*(x[3:6]-x[6:9])/(np.linalg.norm(x[3:6]-x[6:9])**3) + mass[0]*(x[3:6]-x[0:3])/(np.linalg.norm(x[3:6]-x[0:3])**3))
        force[6:9] = -self.G*(mass[0]*(x[6:9]-x[0:3])/(np.linalg.norm(x[6:9]-x[0:3])**3) + mass[1]*(x[6:9]-x[3:6])/(np.linalg.norm(x[6:9]-x[3:6])**3))
        
    def set_random_state(self,x,v):
        '''Draw position and velocity from a normal distribution
        :arg x: cartesian coordinates of the 3 bodies (9-dimensional array)
        :arg v: velocities of the three bodies (9-dimensional array)
        '''

        x[0:9] = np.random.normal(0,1,(9))   
        v[0:9] = np.random.normal(0,1,(9)) 

    def energy(self,x,v):
        '''Compute total energy of the three bodies

        :arg x: cartesian coordinates of the 3 bodies (9-dimensional array)
        :arg v: velocities of the three bodies (9-dimensional array)
        '''

        mass = self.mass
        
        return -self.G*((mass[0]*mass[1])/np.linalg.norm(x[0:3]-x[3:6]) + (mass[1]*mass[2])/np.linalg.norm(x[3:6]-x[6:9]) + (mass[2]*mass[0])/np.linalg.norm(x[6:9]-x[0:3])) \
            + 0.5*mass[0]*np.linalg.norm(v[0:3])**2 + 0.5*mass[1]*np.linalg.norm(v[3:6])**2 + 0.5*mass[2]*np.linalg.norm(v[6:9])**2

class DoublePendulum(DynamicalSystem):
    def __init__(self, mass, L1, L2, g):
        '''2-dimensional double pendulum described by the equations
        of motion

        dx_0/dt = v_0, dv_0/dt = [F1,F2] in scaled force section

        :arg mass: Particle mass
        :arg g: gravitional force constant
        :arg L1: length of first segment of double pendulum
        :arg L2: length of second segment of double pendulum
        '''
        super().__init__(2,mass)
        self.g = g
        self.L1 = L1
        self.L2 = L2

    def compute_scaled_force(self,x,v,force):
        '''Set the entry force[0] of the force vector

        :arg x: angles of bobs wrt vertical (2-dimensional array)
        :arg force: Resulting force vector (2-dimensional array)
        '''
        
        L1 = self.L1
        L2 = self.L2
        mass = self.mass
        g = self.g
        
        mu = 1 + mass[0] + mass[1]
            
        force[0] = (1/(L1*(mu - (math.cos(x[0]-x[1])**2)))) \
                    * (g*(math.sin(x[1])*math.cos(x[0]-x[1])-mu*math.sin(x[0])) \
                       - (L2*(v[1]**2) + L1*(v[0]**2)*math.cos(x[0]-x[1]))*math.sin(x[0]-x[1]))
        force[1] = (1/(L2*(mu - (math.cos(x[0]-x[1])**2)))) \
            * (g*mu*(math.sin(x[0])*math.cos(x[0]-x[1])-math.sin(x[1])) \
               + (L1*mu*(v[0]**2) + L2*(v[1]**2)*math.cos(x[0]-x[1]))*math.sin(x[0]-x[1]))  
        
    def set_random_state(self,x,v):
        '''Draw position and angular velocity from a normal distribution
        :arg x: Angles with vertical (2-dimensional array)
        :arg v: Angular velocities (2-dimensional array)
        '''
                
        x[0:2] = np.random.normal(0,(math.pi)/2,(2)) #angles of mass 1 and 2
        v[0:2] = np.random.normal(0,1,(2)) #angular velocities of mass 1 and 2

    def energy(self,x,v):
        '''Compute total energy E = 1/2*m*v_0^2 + 1/2*k*x_0^2

        :arg x: Angles with vertical (2-dimensional array)
        :arg v: Angular velocities(2-dimensional array)
        '''
        
        L1 = self.L1
        L2 = self.L2
        g = self.g
        mass = self.mass
        
        '''Potential Energy'''
        V = mass[0]*g*L1*(1-math.cos(x[0])) \
            + mass[1]*g*(L1*(1-math.cos(x[0])) \
            + L2*(1-math.cos(x[1])))
        
        '''Kinetic Energy'''
        K = 0.5*mass[0]*(L1**2)*(v[0]**2) \
            + 0.5*mass[1]*(L1**2)*(v[0]**2) \
            + 0.5*mass[1]*(L2**2)*(v[1]**2) \
            + mass[1]*L1*L2*math.cos(x[0]-x[1])*v[0]*v[1]
        
        return V + K 



class FastHarmonicOscillator(DynamicalSystem):
    def __init__(self,mass,k_spring):
        '''One-dimensional harmonic oscillator described by the equations
        of motion

        dx_0/dt = v_0, dv_0/dt = -k/m_0*x_0

        :arg mass: Particle mass
        :arg k_spring: Spring constant k
        '''
        super().__init__(1,mass)
        self.k_spring = k_spring
        # C-code snipped for computing the acceleration update
        self.acceleration_update_code = string.Template('''
        a[0] += -($KSPRING/$MASS)*x[0];
        ''').substitute(KSPRING=self.k_spring,MASS=self.mass)

    def compute_scaled_force(self,x,v,force):
        '''Set the entry force[0] of the force vector
        to -k/m_0*x_0

        :arg x: Particle position x (1-dimensional array)
        :arg force: Resulting force vector (1-dimensional array)
        '''
        force[0] = -self.k_spring/self.mass*x[0]

    def set_random_state(self,x,v):
        '''Draw position and velocity from a normal distribution
        :arg x: Positions (d-dimensional array)
        :arg v: Velocities (d-dimensional array)
        '''
        x[0] = np.random.normal(0,1)
        v[0] = np.random.normal(0,1)

    def energy(self,x,v):
        '''Compute total energy E = 1/2*m*v_0^2 + 1/2*k*x_0^2

        :arg x: Positions (d-dimensional array)
        :arg v: Velocities (d-dimensional array)
        '''
        return 0.5*self.mass*v[0]**2 + 0.5*self.k_spring*x[0]**2
    
    def forward_map(self,x0,v0,t):
        '''Exact forward map
        
        Compute position x(t) and velocity v(t), given initial position x(0) and velocity v(0).
        
        For this use:
        
        x(t) = x(0)*cos(omega*t) + omega*v(0)*sin(omega*t)
        v(t) = -x(0)/omega*sin(omega*t) + v(0)*cos(omega*t)
        
        with omegae = sqrt(k/m)
        
        :arg x0: initial position x(0)
        :arg v0: initial velocity v(0)
        :arg t: final time
        '''
        omega = np.sqrt(self.k_spring/self.mass)
        cos_omegat = np.cos(omega*t)
        sin_omegat = np.sin(omega*t)
        x = np.array(x0[0]*cos_omegat + v0[0]/omega*sin_omegat)
        v = np.array(-x0[0]*omega*sin_omegat + v0[0]*cos_omegat)
        return x, v



class FastDoublePendulum(DynamicalSystem):
    def __init__(self, mass, L1, L2, g=9.81):
        '''2-dimensional double pendulum described by the equations
        of motion

        dx_0/dt = v_0, dv_0/dt = [F1,F2] in scaled force section

        :arg mass: Particle mass
        :arg g: gravitional force constant
        :arg L1: length of first segment of double pendulum
        :arg L2: length of second segment of double pendulum
        '''
        super().__init__(2,mass)
        self.g = g
        self.L1 = L1
        self.L2 = L2
        # C-code snipped for computing the acceleration update
        self.acceleration_header_code = '''
        #include "math.h"
        '''
        self.acceleration_preamble_code = '''
        double cos_x0_x1;
        double sin_x0_x1;
        double sin_x0;
        double sin_x1;
        '''
        self.acceleration_update_code = '''
        cos_x0_x1 = cos(x[0]-x[1]);
        sin_x0_x1 = sin(x[0]-x[1]);
        sin_x0 = sin(x[0]);
        sin_x1 = sin(x[1]);
        a[0] += (1/({L1}*({mu} - (cos_x0_x1*cos_x0_x1))))
             * ({g}*(sin_x1*cos_x0_x1-{mu}*sin_x0)
             - ({L2}*(v[1]*v[1]) + {L1}*(v[0]*v[0])*cos_x0_x1)*sin_x0_x1);
        a[1] += (1/({L2}*({mu} - (cos_x0_x1*cos_x0_x1)))) \
             * ({g}*{mu}*(sin_x0*cos_x0_x1-sin_x1) \
             + ({L1}*{mu}*(v[0]*v[0])+{L2}*(v[1]*v[1])*cos_x0_x1)*sin_x0_x1);
        '''.format(mu=1+self.mass[0]+self.mass[1],L1=self.L1,L2=self.L2,g=self.g)

    def compute_scaled_force(self,x,v,force):
        '''Set the entry force[0] of the force vector

        :arg x: angles of bobs wrt vertical (2-dimensional array)
        :arg force: Resulting force vector (2-dimensional array)
        '''
        L1 = self.L1
        L2 = self.L2
        mass = self.mass
        g = self.g

        mu = 1 + mass[0] + mass[1]

        force[0] = (1/(L1*(mu - (np.cos(x[0]-x[1])**2)))) \
                    * (g*(np.sin(x[1])*np.cos(x[0]-x[1])-mu*np.sin(x[0])) \
                       - (L2*(v[1]**2) + L1*(v[0]**2)*np.cos(x[0]-x[1]))*np.sin(x[0]-x[1]))
        force[1] = (1/(L2*(mu - (np.cos(x[0]-x[1])**2)))) \
            * (g*mu*(np.sin(x[0])*np.cos(x[0]-x[1])-np.sin(x[1])) \
               + (L1*mu*(v[0]**2) + L2*(v[1]**2)*np.cos(x[0]-x[1]))*np.sin(x[0]-x[1]))

    def set_random_state(self,x,v):
        '''Draw position and angular velocity from a normal distribution
        :arg x: Angles with vertical (2-dimensional array)
        :arg v: Angular velocities (2-dimensional array)
        '''

        x[0:2] = np.random.normal(0,(np.pi)/2,(2)) #angles of mass 1 and 2
        v[0:2] = np.random.normal(0,1,(2)) #angular velocities of mass 1 and 2

    def energy(self,x,v):
        '''Compute total energy E = 1/2*m*v_0^2 + 1/2*k*x_0^2

        :arg x: Angles with vertical (2-dimensional array)
        :arg v: Angular velocities(2-dimensional array)
        '''

        L1 = self.L1
        L2 = self.L2
        g = self.g
        mass = self.mass

        '''Potential Energy'''
        V_pot = mass[0]*g*L1*(1-np.cos(x[0])) \
              + mass[1]*g*(L1*(1-np.cos(x[0])) \
              + L2*(1-np.cos(x[1])))

        '''Kinetic Energy'''
        T_kin = 0.5*mass[0]*(L1**2)*(v[0]**2) \
              + 0.5*mass[1]*(L1**2)*(v[0]**2) \
              + 0.5*mass[1]*(L2**2)*(v[1]**2) \
              + mass[1]*L1*L2*np.cos(x[0]-x[1])*v[0]*v[1]

        return V_pot + T_kin


class HenonHeiles(DynamicalSystem):
    def __init__(self,mass):
        '''Two-dimensional Henon-Heiles system with lambda = 1 described by the equations
        of motion

        dx_0/dt = v_0, dv_0/dt = [-x-2xy , -y-x^2+y^2]

        :arg mass: Particle mass
        '''
        
        super().__init__(2,mass)

    def compute_scaled_force(self,x,v,force):
        '''Set the entry force[0] of the force vector
        to 

        :arg x: Particle position x (2-dimensional array)
        :arg force: Resulting force vector (2-dimensional array)
        '''
        force[0] = -(x[0] + 2*x[0]*x[1])/self.mass
        force[1] = -(x[1] + x[0]**2 - x[1]**2)/self.mass

    def set_random_state(self,x,v):
        '''Draw position and velocity from a normal distribution
        :arg x: Positions (2-dimensional array)
        :arg v: Velocities (2-dimensional array)
        arg mass: Mass
        '''
        x[0:2] = np.random.normal((0,1),2)
        v[0:2] = np.random.normal((0,1),2)
        
    def energy(self,x,v):
        '''Compute total energy E = V(x,y) + 0.5((mx)^2 + (my)^2)

        :arg x: Positions (2-dimensional array)
        :arg v: Velocities (2-dimensional array)
        '''
        
        mass = self.mass
        
        V = 0.5*(x[0]**2 + x[1]**2 + 2*(x[0]**2)*x[1] - (2/3)*x[1]**3)
        
        K = 0.5*(mass*(v[0])**2 + mass*(v[1])**2)
        
        return V+K
