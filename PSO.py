#  Python Code for Particle Swarm Optimization (PSO)
#  Author: Arshad Afzal, IIT Kanpur, India
#  For Questions/ Comments, please email to arshad.afzal@gmail.com
import numpy as np
import matplotlib.pyplot as plt
from math import *
from hart import *
#  Input Parameters for Optimization
n = int(input('Enter number of Design Variables :'))
print("Enter lower bound on Design Variables :")
a = []
for i in range(n):
    x = float(input())
    a.append(x)
lb = np.zeros([n, 1])
for i in range(n):
    lb[i] = a[i]
print("Enter upper bound on Design Variables :")
b = []
for i in range(n):
    x = float(input())
    b.append(x)
# b = [i for i in input().split()]
ub = np.zeros([n, 1])
for i in range(n):
    ub[i] = b[i]
pop = int(input('Enter Population Size (Swarm size) :'))
# Convergence Criterion
maxiter = int(input('Enter Maximum number of Iterations:'))
funtol = float(input('Enter the Relative Tolerance in Function:'))
#  PSO Model Co-efficients (Clerc and Kennedy, 2002)
kappa = 1
phi1 = 2.05
phi2 = 2.05
phi = phi1 + phi2
chi = 2 * kappa/abs(2 - phi - sqrt(pow(phi, 2) - 4 * phi))
w = chi
wdamp = 0.9
c1 = chi * phi1
c2 = chi * phi2
#  Initialization
showiterinfo = True
maxvelocity = 0.2 * (np.ones([1, n]) - np.zeros([1, n]))
minvelocity = - maxvelocity
x = np.zeros([pop, n])
optimum = np.zeros([1, n])
particle_pos = np.random.uniform(0, 1, [pop, n])
particle_vel = np.random.uniform(0, 1, [pop, n])
particle_cost = np.zeros([pop, 1])
best_cost = np.zeros([1, 1])
#  Convert Particle Position to Original  System for function Evaluation
for i in range(pop):
    for j in range(n):
        x[i][j] = (ub[j] - lb[j]) * particle_pos[i][j] + lb[j]
for i in range(pop):
    particle_cost[i] = hartmann3(x[i:i+1, 0:n])  # Problem Dependent Function Name
particle_bestpos = particle_pos
particle_bestcost = particle_cost
# Global Best
best_cost = particle_bestcost.min()
j = particle_bestcost.argmin()
globalbest = particle_bestpos[j:j+1, 0:n]
t = best_cost
#  Create File to Write Output
f = open("Outputfile.txt", "a")
#  Create Lists for Plots
x_data = []  # Iterations
y_data = []  # Best Cost Function
z_data = []  # Average Cost Function
#  Main Loop
for it in range(maxiter):
    for i in range(pop):
        #  Particle Velocity Update
        particle_vel[i:i+1, 0:n] = w * particle_vel[i:i+1, 0:n]\
         + c1 * np.multiply(np.random.uniform(0, 1, [1, n]), (particle_bestpos[i:i+1, 0:n]-particle_pos[i:i+1, 0:n]))\
         + c2 * np.multiply(np.random.uniform(0, 1, [1, n]), (globalbest - particle_pos[i:i+1, 0:n]))
        #  Check Bounds on Particle Velocity
        particle_vel[i:i + 1, 0:n] = np.maximum(particle_vel[i:i+1, 0:n], minvelocity)
        particle_vel[i:i + 1, 0:n] = np.minimum(particle_vel[i:i+1, 0:n], maxvelocity)
        #  Particle Position Update
        particle_pos[i:i+1, 0:n] = particle_pos[i:i+1, 0:n] + particle_vel[i:i+1, 0:n]
        #  Check Bounds on Particle Position
        particle_pos[i:i + 1, 0:n] = np.maximum(particle_pos[i:i+1, 0:n], np.zeros([1, n]))
        particle_pos[i:i + 1, 0:n] = np.minimum(particle_pos[i:i+1, 0:n], np.ones([1, n]))
        #  Convert Particle Position to Original System for function Evaluation
        for j in range(n):
            x[i][j] = (ub[j] - lb[j]) * particle_pos[i][j] + lb[j]
        particle_cost[i:i + 1, 0:n] = hartmann3(x[i:i + 1, 0:n])  # Problem Dependent Function Name
        #  Personal and Global Best
        if particle_cost[i] < particle_bestcost[i]:
            particle_bestpos[i:i + 1, 0:n] = particle_pos[i:i+1, 0:n]
            particle_bestcost[i] = particle_cost[i]
    best_cost = particle_bestcost.min()
    j = particle_bestcost.argmin()
    globalbest = particle_bestpos[j:j + 1, 0:n]
    avgbest_cost = np.mean(particle_bestcost)
    err_f = abs((best_cost - t) / best_cost)
    #  Display Best Cost with Iteration
    if showiterinfo:
        print("Iteration: " + str(it) + " Best Cost: " + str(best_cost))
    w = w * wdamp
    t = best_cost
    x_data.append(it + 1)
    y_data.append(best_cost)
    z_data.append(avgbest_cost)
    if err_f < funtol:
        print("\nAlgorithm Stopped: Relative change in Function Less Than Specified Tolerance")
        f.write("\nAlgorithm Stopped: Relative change in Function Less Than Specified Tolerance\n")
        break
    if it == maxiter - 1:
        print("\nAlgorithm Stopped: Maximum Number of Iterations Reached")
        f.write("\nAlgorithm Stopped: Maximum Number of Iterations Reached\n")
        break
#  Optimum Particle Position (globalbest) to Original System
    for j in range(n):
        optimum[0][j] = (ub[j] - lb[j]) * globalbest[0][j] + lb[j]
#  Print OUTPUT
print("\nOptimum Solution: " + str(optimum))
f.write("Optimum Solution:")
np.savetxt(f, optimum, fmt="%2.4f", delimiter='  ')
print("\nFunction Value: " + str(best_cost))
f.write("Function Value:")
f.write(str(best_cost))
f.close()
#  Plot for Best and Average Cost Function with Iterations
plt.plot(x_data, y_data, '-b')
plt.plot(x_data, z_data, '-r')
plt.xlabel('Iteration')
plt.ylabel("Cost Function")
plt.grid()
plt.legend(labels=('Best Cost', 'Average Cost'), loc='upper right', frameon=False)
plt.show()

