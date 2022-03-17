"""
Versions of Gradient Temporal Difference Learning
Donghwan Lee, Han-Dong Lim, Jihoon Park, and Okyong Choi
"""

from mdp import MDP
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

class LinApprox(MDP):
    def __init__(self, state_size = 100, action_size = 10, feature_vector_size = 10): 
        super().__init__(state_size, action_size, feature_vector_size)
        self.phi = self.make_feature_func()

        # solution = -inv(phi'*D*(gamma*P_target - I(state_size))*phi)*Phi'*D*reward;
        self.sol = -np.linalg.inv(self.phi.T@self.D@(self.gamma*self.P_target - np.eye(self.state_size))@self.phi)@self.phi.T@self.D@self.reward

    def make_feature_func(self):
        # Make sure the feature function sparse
        phi = 1-2*np.random.rand(1, self.state_size)
        #while(np.linalg.matrix_rank(phi) < self.feature_vector_size):
        for i in range(1, self.feature_vector_size):
            vec = 1-2*np.random.rand(1, self.state_size)
            phi = np.append(phi, vec, axis = 0)
        return phi.transpose()


if __name__ == '__main__':
    mdp = MDP()
    
    #GTD2 parameters
    theta1 = np.random.rand(mdp.feature_vector_size, 1)
    lambda1 = np.random.rand(mdp.feature_vector_size, 1)
    
    #GTD3 parameters
    theta2 = np.random.rand(mdp.feature_vector_size, 1)
    lambda2 = np.random.rand(mdp.feature_vector_size, 1)

    #GTD4 parameters
    theta3 = np.random.rand(mdp.feature_vector_size, 1)            
    lambda3 = np.random.rand(mdp.feature_vector_size, 1)

    steps = 100000
    error_vec1 = np.zeros(steps)
    error_vec2 = np.zeros(steps)
    error_vec3 = np.zeros(steps)
    for step in range(steps):
        #Generates a random variable in 1, 2, ..., n given a prob distribution 
        state = np.random.choice(mdp.state_size, 1, p = mdp.d)
        state = state[0]
        action = np.random.choice(mdp.action_size, 1, p = mdp.beta[state])
        action = action[0]
        next_state = np.random.choice(mdp.state_size, 1, p = mdp.P_beta[:,state])
        next_state = next_state[0]

        # Importance sampling ratio
        rho = mdp.target[state][action]/mdp.beta[state][action]
        
        # Diminishing step size
        step_size = 10/(step+100)

        # GTD (off-policy)
        delta = rho*mdp.reward[state] + mdp.gamma*rho*mdp.phi[next_state]@theta1 - mdp.phi[state]@theta1
        theta1 = theta1 + step_size * (mdp.phi[state].reshape(-1,1) - mdp.gamma*rho*mdp.phi[next_state].reshape(-1,1)) * mdp.phi[state]@lambda1
        lambda1 = lambda1 + step_size * (delta - mdp.phi[state]@lambda1) * mdp.phi[state].reshape(-1,1)

        # GTD3
        delta = rho*mdp.reward[state] + mdp.gamma*rho*mdp.phi[next_state]@theta2 - mdp.phi[state]@theta2
        theta2 = theta2 + step_size * ((mdp.phi[state].reshape(-1,1) - mdp.gamma*rho*mdp.phi[next_state].reshape(-1,1)) * mdp.phi[state]@lambda2 - mdp.phi[state].reshape(-1,1)*mdp.phi[state]@theta2)
        lambda2 = lambda2 + step_size * delta * mdp.phi[state].reshape(-1,1)

        # GTD4
        sigma1 = 100/(steps+1000)
        delta = rho*mdp.reward[state] + mdp.gamma*rho*mdp.phi[next_state]@theta3 - mdp.phi[state]@theta3
        theta3 = theta3 + step_size * ((mdp.phi[state].reshape(-1,1) - mdp.gamma*rho*mdp.phi[next_state].reshape(-1,1)) * mdp.phi[state]@lambda3 - sigma1*mdp.phi[state].reshape(-1,1)*mdp.phi[state]@theta3)
        lambda3 = lambda3 + step_size * (delta - mdp.phi[state]@lambda3) * mdp.phi[state].reshape(-1,1)


        error1 = np.linalg.norm(mdp.sol-theta1 ,2)
        error2 = np.linalg.norm(mdp.sol-theta2, 2)
        error3 = np.linalg.norm(mdp.sol-theta3, 2)

        error_vec1[step] = error1
        error_vec2[step] = error2
        error_vec3[step] = error3
    
    plt.plot(error_vec1, 'b', label = 'GTD2')
    plt.plot(error_vec2, 'r', label = 'GTD3')
    plt.plot(error_vec3, 'g', label = 'GTD4')
    plt.legend()
    plt.yscale("log")
    plt.savefig('result.png')