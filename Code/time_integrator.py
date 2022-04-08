import numpy as np
import subprocess
import string
import ctypes
import hashlib


class TimeIntegrator(object):
    def __init__(self,dynamical_system,dt):
        '''Abstract base class for a single step traditional time integrator

        :arg dynamical_system: Dynamical system to be integrated
        :arg dt: time step size
        '''
        self.dynamical_system = dynamical_system
        self.dt = dt
        self.x = np.zeros(dynamical_system.dim)
        self.v = np.zeros(dynamical_system.dim)
        self.force = np.zeros(dynamical_system.dim)

    def set_state(self,x,v):
        '''Set the current state of the integrator to a specified
        position and velocity.

        :arg x: New position vector
        :arg v: New velocity vector
        '''
        self.x[:] = x[:]
        self.v[:] = v[:]
        self.dynamical_system.compute_scaled_force(self.x,self.v,self.force)

    def integrate(self,n_steps):
        '''Carry out n_step timesteps, starting from the current set_state
        and updating this

        :arg steps: Number of integration steps
        '''
        pass
    
    def energy(self):
        '''Return the energy of the underlying dynamical system for
        the current position and velocity'''
        return self.dynamical_system.energy(self.x,self.v)

class ForwardEulerIntegrator(TimeIntegrator):
    def __init__(self,dynamical_system,dt):
        '''Forward Euler integrator given by

        x_j^{(t+dt)} = x_j^{(t)} + dt*v_j
        v_j^{(t+dt)} = v_j^{(t)} + dt*F_j(x^{(t)})/m_j

        :arg dynamical_system: Dynamical system to be integrated
        :arg dt: time step size
        '''
        super().__init__(dynamical_system,dt)

    def integrate(self,n_steps):
        '''Carry out n_step timesteps, starting from the current set_state
        and updating this

        :arg steps: Number of integration steps
        '''
        for k in range(n_steps):
            self.x[:] += self.dt*self.v[:]
            self.v[:] += self.dt*self.force[:]
            # Compute force at next timestep
            self.dynamical_system.compute_scaled_force(self.x,self.v,self.force)

class VerletIntegrator(TimeIntegrator):
    def __init__(self,dynamical_system,dt):
        '''Verlet integrator given by

        x_j^{(t+dt)} = x_j^{(t)} + dt*v_j + dt^2/2*F_j(x^{(t)})/m_j
        v_j^{(t+dt)} = v_j^{(t)} + dt^2/2*(F_j(x^{(t)})/m_j+F_j(x^{(t+dt)})/m_j)

        :arg dynamical_system: Dynamical system to be integrated
        :arg dt: time step size
        '''
        super().__init__(dynamical_system,dt)

    def integrate(self,n_steps):
        '''Carry out n_step timesteps, starting from the current set_state
        and updating this

        :arg steps: Number of integration steps
        '''
        for k in range(n_steps):
            self.x[:] += self.dt*self.v[:] + 0.5*self.dt**2*self.force[:]
            self.v[:] += 0.5*self.dt*self.force[:]
            self.dynamical_system.compute_scaled_force(self.x,self.v,self.force)
            self.v[:] += 0.5*self.dt*self.force[:]

class SymplecticEulerIntegrator(TimeIntegrator):
    def __init__(self,dynamical_system,dt):
        '''Symplectic Euler integrator given by

        v_j^{(t+dt)} = v_j^{(t)} + dt*F_j(x^{(t)})/m_j
        x_j^{(t+dt)} = x_j^{(t)} + dt*v_j^{(t+dt)}

        :arg dynamical_system: Dynamical system to be integrated
        :arg dt: time step size
        '''
        super().__init__(dynamical_system,dt)

    def integrate(self,n_steps):
        '''Carry out n_step timesteps, starting from the current set_state
        and updating this

        :arg steps: Number of integration steps
        '''
        for k in range(n_steps):
            self.v[:] += self.dt*self.force[:]
            self.x[:] += self.dt*self.v[:]
            # Compute force at next timestep
            self.dynamical_system.compute_scaled_force(self.x,self.v,self.force)
            
class RK4Integrator(TimeIntegrator):
    def __init__(self,dynamical_system,dt):
        '''Fourth order Runge-Kutta integrator given by

        x_j^{(t+dt)} = x_j^{(t)} + dt/6 (vZ_1 + 2vZ_2 + 2vZ_3 +vZ_4)
        v_j^{(t+dt)} = v_j^{(t)} + dt/6 (aZ_1 + 2aZ_2 + 2aZ_3 +aZ_4)

        :arg dynamical_system: Dynamical system to be integrated
        :arg dt: time step size
        '''
        super().__init__(dynamical_system,dt)
        
        self.xZ_1 = np.zeros(dynamical_system.dim)
        self.xZ_2 = np.zeros(dynamical_system.dim)
        self.xZ_3 = np.zeros(dynamical_system.dim)
        self.xZ_4 = np.zeros(dynamical_system.dim)
        self.vZ_1 = np.zeros(dynamical_system.dim)
        self.vZ_2 = np.zeros(dynamical_system.dim)
        self.vZ_3 = np.zeros(dynamical_system.dim)
        self.vZ_4 = np.zeros(dynamical_system.dim)
        
    def integrate(self,n_steps):
        '''Carry out n_step timesteps, starting from the current set_state
        and updating this

        :arg steps: Number of integration steps
        '''
        for k in range(n_steps):
                       
            self.vZ_1[:] = self.v[:]
            self.xZ_1[:] = self.x[:]
            
            self.dynamical_system.compute_scaled_force(self.xZ_1,self.vZ_1,self.force)
            
            self.vZ_2[:] = self.v[:] + 0.5*self.dt*self.force
            self.xZ_2[:] = self.x[:] + 0.5*self.dt*self.vZ_1[:]
            
            self.v[:] += self.dt/6 * self.force
            
            self.dynamical_system.compute_scaled_force(self.xZ_2,self.vZ_2,self.force)
            
            self.vZ_3[:] = self.v[:] + 0.5*self.dt*self.force
            self.xZ_3[:] = self.x[:] + 0.5*self.dt*self.vZ_2[:]
            
            self.v[:] += self.dt/6 * self.force

            self.dynamical_system.compute_scaled_force(self.xZ_3,self.vZ_3,self.force)           
            
            self.vZ_4[:] = self.v[:] + self.dt*self.force
            self.xZ_4[:] = self.x[:] + self.dt*self.vZ_3[:]
            
            self.v[:] += self.dt/6 * self.force
            
            self.dynamical_system.compute_scaled_force(self.xZ_4,self.vZ_4,self.force)
            
            self.v[:] += self.dt/6 * self.force        
            self.x[:] += self.dt/6 * (self.vZ_1 + 2*self.vZ_2 + 2*self.vZ_3 + self.vZ_4)
                       
            # Compute force at next timestep


class ExactIntegrator(TimeIntegrator):
    def __init__(self,dynamical_system,dt):
        '''Exact integrator
        
        Integrate the equations of motion exactly, if the dynamical system supports this.

        :arg dynamical_system: Dynamical system to be integrated
        :arg dt: time step size
        '''
        super().__init__(dynamical_system,dt)
        self.label = 'Exact'

    def integrate(self,n_steps):
        '''Carry out n_step timesteps, starting from the current set_state
        and updating this

        :arg steps: Number of integration steps
        '''
        self.x[:], self.v[:] = self.dynamical_system.forward_map(self.x[:],self.v[:],n_steps*self.dt)
        
        

class FastVerletIntegrator(TimeIntegrator):
    def __init__(self,dynamical_system,dt):
        '''Verlet integrator given by

        x_j^{(t+dt)} = x_j^{(t)} + dt*v_j + dt^2/2*F_j(x^{(t)})/m_j
        v_j^{(t+dt)} = v_j^{(t)} + dt^2/2*(F_j(x^{(t)})/m_j+F_j(x^{(t+dt)})/m_j)

        :arg dynamical_system: Dynamical system to be integrated
        :arg dt: time step size
        '''
        super().__init__(dynamical_system,dt)
        self.label = 'Verlet'
        # Check whether dynamical system has a C-code snippet for updating the acceleration
        self.fast_code = hasattr(self.dynamical_system,'acceleration_update_code')
        # If this is the case, auto-generate fast C code for the Velocity Verlet update
        if self.fast_code:
            c_sourcecode = string.Template('''
            void velocity_verlet(double* x, double* v, int nsteps) {
                double a[$DIM];
                for (int k=0;k<nsteps;++k) {
                    for (int j=0;j<$DIM;++j) a[j] = 0;
                    $ACCELERATION_UPDATE_CODE
                    for (int j=0;j<$DIM;++j) {
                        x[j] += $DT*v[j] + 0.5*$DT*$DT*a[j];
                    }
                    $ACCELERATION_UPDATE_CODE
                    for (int j=0;j<$DIM;++j) {
                        v[j] += 0.5*$DT*a[j];
                    }
                }
            }
            ''').substitute(DIM=self.dynamical_system.dim,
                            DT=self.dt,
                            ACCELERATION_UPDATE_CODE=self.dynamical_system.acceleration_update_code)
            sha = hashlib.md5()
            sha.update(c_sourcecode.encode())
            filestem = './velocity_verlet_'+sha.hexdigest()
            so_file = filestem+'.so'
            source_file = filestem+'.c'
            with open(source_file,'w') as f:
                print (c_sourcecode,file=f)
            # Compile source code (might have to adapt for different compiler)
            subprocess.run(['gcc',
                            '-fPIC','-shared','-o',
                            so_file,
                            source_file])
            self.c_velocity_verlet = ctypes.CDLL(so_file).velocity_verlet
            self.c_velocity_verlet.argtypes = [np.ctypeslib.ndpointer(ctypes.c_double,
                                               flags="C_CONTIGUOUS"),
                                               np.ctypeslib.ndpointer(ctypes.c_double,
                                               flags="C_CONTIGUOUS"),
                                               np.ctypeslib.c_intp]

    def integrate(self,n_steps):
        '''Carry out n_step timesteps, starting from the current set_state
        and updating this

        :arg steps: Number of integration steps
        '''
        if self.fast_code:
            self.c_velocity_verlet(self.x,self.v,n_steps)
        else:
            for k in range(n_steps):
                self.x[:] += self.dt*self.v[:] + 0.5*self.dt**2*self.force[:]
                self.dynamical_system.apply_constraints(self.x)
                self.v[:] += 0.5*self.dt*self.force[:]
                self.dynamical_system.compute_scaled_force(self.x,self.v,self.force)
                self.v[:] += 0.5*self.dt*self.force[:]