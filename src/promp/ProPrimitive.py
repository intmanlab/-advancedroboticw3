import numpy as np
import phase as phase
import basis as basis
import promps as promps
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numbers import Number
from matplotlib.lines import Line2D



Xee = list()
Yee = list()
Zee = list()

coord_s1 = []
tf = []  # create a list()

# cd to promp folder
with open('./demos/demos.npz', 'r') as f:
    Q = np.load(f) # Q shape is (121, 162, 7), 3D array, i = 121 is demo length, 162 is samples length and 7 is dof length
    Q = Q['data']
    #print(Q[0])
    print('Q length:',len(Q))


print('Q=', Q.shape)
sdemo = Q.shape[0] # nb of demos
ssamples = Q.shape[1]  # nb os samples per demo
print('ssamples', ssamples)
sdof = Q.shape[2] # nb of dofs per demo per 
print('sdof', sdof)


## random time vector, since we didn't collected data
tff1 = np.linspace(0,1, 162)
tff1 = np.repeat(np.array([tff1]), sdemo, axis = 0)


################################################
#To plot demonstrated end-eff trajectories
def plotEE():
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    for i in range(0,len(Q)): # demos, 
        endEffTraj = Q[i] # 1 demo
        Xee.append(endEffTraj[:,0])
        Yee.append(endEffTraj[:,1])
        Zee.append(endEffTraj[:,2])
        x_ee = endEffTraj[:,0] / 1000
        y_ee = endEffTraj[:,1] / 1000
        z_ee = endEffTraj[:,2] / 1000
        ax.scatter(endEffTraj[:,0], endEffTraj[:,1], endEffTraj[:,2], c='b', marker='.') #X, Y , Z
    plt.title('EndEff')
    plt.show()
    

plotEE()


################################################################
phaseGenerator = phase.LinearPhaseGenerator() # generates z = z_dot *time, a constructor of the class LinearPhaseGenerator
basisGenerator = basis.NormalizedRBFBasisGenerator(phaseGenerator, numBasis= 4, duration=1, basisBandWidthFactor=1,
                                                   numBasisOutside=1)  
time_normalised = np.linspace(0, 1, 100)  # 1sec duration 
nDof = 1
plotDof = 1


##################################################
# Learnt promp in Task Space 

learnedProMP1 = promps.ProMP(basisGenerator, phaseGenerator, nDof)
learner1 = promps.MAPWeightLearner(learnedProMP1)  # Initialization 

learntTraj1Xee = learner1.learnFromXDataTaskSapce(Q[:,:,0]/1000, tff1)
traj1_Xee = learnedProMP1.getTrajectorySamplesX(time_normalised, 1) # get samples from the ProMP in the joint space 
learntTraj1Yee = learner1.learnFromYDataTaskSapce(Q[:,:,1]/1000, tff1)
traj1_Yee = learnedProMP1.getTrajectorySamplesY(time_normalised, 1) 
learntTraj1Zee = learner1.learnFromZDataTaskSapce(Q[:,:,2]/1000, tff1)
traj1_Zee = learnedProMP1.getTrajectorySamplesZ(time_normalised, 1)

##################################################

start = []
IC0 = np.array([0, 0, 1]) 
start.append(IC0)
mu_x_IC = start[0] 


## Fill-in a desired object goal position in task space  (To Do: Task 1)
goal = np.array([..., ..., ...])
 

#####################################################################################################################################
# Visualize if the learnt promp passes thru the desired goal (To Do: Task 2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(.... , .... , .... , c='b', marker='.')
ax.scatter(mu_x_IC[0], mu_x_IC[1], mu_x_IC[2], s = 100, c='y', marker='o')
ax.scatter(traj1_Xee[-1], traj1_Yee[-1], traj1_Zee[-1], s = 100, c='r', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Task space learnt promp1')
plt.show()


#######################################################################################################################################

# Conditioning at 1st goal point : Task 3

mu_x_tf = goal 
sig_x_tf = np.eye(3) * 0.0
print('mu_x_tf=', mu_x_tf)
#### cond at t0
mu_x_t0 = start[0] 
sig_x_t0 = np.eye(3) * 0.0


traj_conditioned_tf = learnedProMP1.taskSpaceConditioning(.... , ..... , ....) # To Do
traj_Xee_condT = traj_conditioned_tf.getTrajectorySamplesX(time_normalised, 1)
traj_Yee_condT = traj_conditioned_tf.getTrajectorySamplesY(time_normalised, 1)
traj_Zee_condT = traj_conditioned_tf.getTrajectorySamplesZ(time_normalised, 1)



## Plot the generated promp (Task 4)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(..... , ..... , .... , c='b', marker='.')
ax.scatter(....., .... , ..... , s = 100, c='y', marker='o')
ax.scatter(..... , ..... , ...... , s = 100, c='r', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Task space conditioned promp 1')
plt.show()


# Change the nb and width of basis functions and discuss the observations of the primitive behavior (Task 5)


## Save the generated task space promp (Task 6)
trajectories_task_conditioned = np.array([traj_Xee_condT, traj_Yee_condT,traj_Zee_condT])
# ......

print('Finished basic framework, next week we will work on conditioning at multiple waypoints')



