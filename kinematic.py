import numpy as np
import math

""""
Robot hardware
- Alpha = 90 & -90 
- Beta = 0
- l = 10cm mean 30px
- r = 2.5cm mean 7.5px
- Theta at first is -90 deg
"""
class Forward_kinematic :
    def __init__(self, alpha, beta, l, r, theta, num_wheel, phi):
        self.alpha = alpha
        self.beta = beta
        self.l = l # This mean l =7.5 
        self.r = r # r = 0.83
        self.theta = theta
        self.num_wheel = num_wheel
        self.phi = phi
    
    
    def rotation_matrix(self):
        
        rotation_matrix = np.zeros((3,3))

        # ----------- Convert to degree ---------------------
        rotation_matrix[0][0] = math.cos(math.radians(self.theta))
        rotation_matrix[0][1] = - math.sin(math.radians(self.theta))
        rotation_matrix[1][0] = math.sin(math.radians(self.theta))
        rotation_matrix[1][1] = math.cos(math.radians(self.theta))
        rotation_matrix[2][2] = 1
        """
        
        # Conver radians
        rotation_matrix[0][0] = math.cos(self.theta)
        rotation_matrix[0][1] = - math.sin(self.theta)
        rotation_matrix[1][0] = math.sin(self.theta)
        rotation_matrix[1][1] = math.cos(self.theta)
        rotation_matrix[2][2] = 1
        """

        #print(matrix_inverse)
        return rotation_matrix
    
    
    """
    def J1f_inverse(self):
        j1f = np.zeros((self.num_wheel,3))
        for colum in range(self.num_wheel):
            j1f[colum][0] = math.sin(math.radians(self.alpha[colum] + self.beta[colum] ))
            j1f[colum][1] = -math.cos(math.radians(self.alpha[colum] + self.beta[colum] ))
            j1f[colum][2] = -self.l * math.cos(math.radians(self.beta[colum]))

        #print(j1f)    

        j1f_inverse = j1f.T @ (np.linalg.pinv(j1f @ j1f.T))
        #print(j1f_inverse)
        #print(self.alpha[0], self.alpha[1])
        return j1f_inverse


    def J2(self):
        j2 = np.zeros((self.num_wheel,self.num_wheel))
        for i in range(self.num_wheel):
            j2[i][i] = self.r

        #print(j2)
        return j2
    """
    def matrix_2(self):    
        mtrx_2 = np.zeros((3,1))
        mtrx_2[0][0] = ( self.r * self.phi[0] + self.r * self.phi[1] ) / 2
        mtrx_2[1][0] = 0
        mtrx_2[2][0] = self.r * self.phi[0] + (- self.r * self.phi[1]) / 2 * self.l 
        print(mtrx_2)
        return mtrx_2
    
    



forward_kine = Forward_kinematic((135, -135), (45, -45), math.sqrt(10**2 + 10**2), 0.025, 90, 2, (15,15))
rotation = forward_kine.rotation_matrix()
mtrx_2 = forward_kine.matrix_2()
# #print(rotation)
# j1f_inv = forward_kine.J1f_inverse()
# #print(j1f_inv)
# j2 = forward_kine.J2()
# phi = np.array([15,15]).reshape(2,1)
# #print(phi)

global_frame = rotation @ mtrx_2
print(global_frame)

print(np.cos(np.pi/2))

