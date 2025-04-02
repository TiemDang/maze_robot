import numpy as np
import math

""""
Robot hardware
- Alpha = 135 & -135 
- Beta = 45 & -45
- l = 10cm mean 30px
- r = 2.5cm mean 7.5px
- Theta at first is 90 deg
"""
class Forward_kinematic :
    def __init__(self, alpha, beta, l, r, theta, num_wheel):
        self.alpha = alpha
        self.beta = beta
        self.l = l # This mean l =7.5 
        self.r = r # r = 0.83
        self.theta = theta
        self.num_wheel = num_wheel
        #self.phi = phi
    
    
    def inv_rotation_matrix(self):
        
        inv_rotation_matrix = np.zeros((3,3))

        
        # # ----------- Convert to degree ---------------------
        # inv_rotation_matrix[0][0] = math.cos(math.radians(self.theta))
        # inv_rotation_matrix[0][1] = - math.sin(math.radians(self.theta))
        # inv_rotation_matrix[1][0] = math.sin(math.radians(self.theta))
        # inv_rotation_matrix[1][1] = math.cos(math.radians(self.theta))
        # inv_rotation_matrix[2][2] = 1
        
        
        # Conver radians
        inv_rotation_matrix[0][0] = math.cos(self.theta)
        inv_rotation_matrix[0][1] = - math.sin(self.theta )
        inv_rotation_matrix[1][0] = math.sin(self.theta)
        inv_rotation_matrix[1][1] = math.cos(self.theta)
        inv_rotation_matrix[2][2] = 1
        

        #print(matrix_inverse)
        return inv_rotation_matrix


    def J1f_inverse(self):
        j1f = np.zeros((self.num_wheel,3))
        for colum in range(self.num_wheel):
            j1f[colum][0] = math.sin(self.alpha[colum] + self.beta[colum] )
            j1f[colum][1] = -math.cos(self.alpha[colum] + self.beta[colum] )
            j1f[colum][2] = -self.l * math.cos(self.beta[colum])
    
        j1f_inverse = np.linalg.pinv(j1f)
        #print(self.alpha[0], self.alpha[1])
        return j1f_inverse


    def J2(self):
        j2 = np.zeros((self.num_wheel,self.num_wheel))
        for i in range(self.num_wheel):
            j2[i][i] = self.r

        return j2

    # def matrix_2(self):    
    #     mtrx_2 = np.zeros((3,1))
    #     mtrx_2[0][0] = ( self.r * self.phi[0] + self.r * self.phi[1] ) / 2
    #     mtrx_2[1][0] = 0
    #     mtrx_2[2][0] = self.r * self.phi[0] + (- self.r * self.phi[1]) / 2 * self.l 
    #     #print(mtrx_2)
    #     return mtrx_2
    
    

class Inverse_kinematic :
    def __init__(self, alpha, beta, l, r, theta, num_wheel):
        self.alpha = alpha
        self.beta = beta
        self.l = l
        self.r = r
        self.num_wheel = num_wheel
        self.theta = theta
    
    def rotation_matrix(self) :
        rotation_matrix = np.zeros((3,3))

        rotation_matrix[0][0] = math.cos(self.theta)
        rotation_matrix[0][1] = math.sin(self.theta)
        rotation_matrix[1][0] = - math.sin(self.theta)
        rotation_matrix[1][1] = math.cos(self.theta)
        rotation_matrix[2][2] = 1

        return rotation_matrix
    
    def J1(self):
        j1f = np.zeros((self.num_wheel,3))
        for colum in range(self.num_wheel):
            j1f[colum][0] = math.sin(self.alpha[colum] + self.beta[colum] )
            j1f[colum][1] = -math.cos(self.alpha[colum] + self.beta[colum] )
            j1f[colum][2] = -self.l * math.cos(self.beta[colum])
        #j1f = np.array([[1, 0, -self.l], [-1, 0, -self.l]])

        return j1f

    def J2_inv(self):
        j2 = np.zeros((self.num_wheel,self.num_wheel))
        for i in range(self.num_wheel):
            j2[i][i] = self.r

        j2 = np.linalg.inv(j2)
        return j2
    
# Forward = Forward_kinematic((np.pi/2, -np.pi/2), (-np.pi/2, np.pi/2), 0.1, 0.05, 0 ,2)
# rtt = Forward.inv_rotation_matrix()
# j1f = Forward.J1f_inverse()
# print(j1f)

