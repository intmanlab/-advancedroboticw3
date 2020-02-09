import matplotlib.pyplot as plt
import numpy as np
import phase as phase
import basis as basis

import scipy.stats as stats


class ProMP:

    def __init__(self, basis, phase, numDoF):
        self.basis = basis
        self.phase = phase
        self.numDoF = numDoF
        self.numBasis = basis.numBasis
        self.numWeights = basis.numBasis * self.numDoF  
        self.muX = np.zeros(self.numWeights) 
        self.covMatX = np.eye(self.numWeights) 
        self.muY = np.zeros(self.numWeights)
        self.covMatY = np.eye(self.numWeights)
        self.muZ = np.zeros(self.numWeights) 
        self.covMatZ = np.eye(self.numWeights)
        self.observationSigma = np.ones(self.numDoF) 



    def getTrajectorySamplesX(self, time, n_samples=1):
        basisMultiDoF = self.basis.basisMultiDoF(time, self.numDoF)
        weightsX = np.random.multivariate_normal(.... , .... , .....)
        weightsX = weightsX.transpose()
        trajectoryFlatX = basisMultiDoF.dot(weightsX)
        trajectoryFlatX = trajectoryFlatX.reshape((self.numDoF, trajectoryFlatX.shape[0] / self.numDoF, n_samples))
        trajectoryFlatX = np.transpose(trajectoryFlatX, (1, 0, 2))
        return trajectoryFlatX


    def getTrajectorySamplesY(self, time, n_samples=1):
        basisMultiDoF = self.basis.basisMultiDoF(time, self.numDoF)
        weightsY = np.random.multivariate_normal(...., ...., ....)
        weightsY = weightsY.transpose()
        trajectoryFlatY = .....
        trajectoryFlatY = trajectoryFlatY.reshape((self.numDoF, trajectoryFlatY.shape[0] / self.numDoF, n_samples))
        trajectoryFlatY = np.transpose(trajectoryFlatY, (1, 0, 2))
        return trajectoryFlatY


    def getTrajectorySamplesZ(self, time, n_samples=1):
        basisMultiDoF = self.basis.basisMultiDoF(time, self.numDoF)
        weightsZ = np.random.multivariate_normal(.... , .... , ....)
        weightsZ = weightsZ.transpose()
        trajectoryFlatZ = ......
        trajectoryFlatZ = trajectoryFlatZ.reshape((self.numDoF, trajectoryFlatZ.shape[0] / self.numDoF, n_samples))
        trajectoryFlatZ = np.transpose(trajectoryFlatZ, (1, 0, 2))
        return trajectoryFlatZ


    def taskSpaceConditioning(self, time, desiredXmean, desiredXVar):
        newProMP = ProMP(self.basis, self.phase, 1)
        basisMatrix = self.basis.basisMultiDoF(time, 1)
        tempX = self.covMatX.dot(basisMatrix.transpose())
        tempY = self.covMatY.dot(basisMatrix.transpose())
        tempZ = self.covMatZ.dot(basisMatrix.transpose())
        LX = np.linalg.solve(desiredXVar[0,0] + .... .dot(tempX), .... )
        LX = LX.transpose()
        LY = np.linalg.solve(desiredXVar[1,1] + .... .dot(tempY), .... )
        LY = LY.transpose()
        LZ = np.linalg.solve(desiredXVar[2,2] + .... .dot(tempZ), .... )
        LZ = LZ.transpose()
        newProMP.muX = .... + LX.dot(desiredXmean[0] - basisMatrix.dot(....))
        newProMP.covMatX = .... - LY.dot(basisMatrix).dot(.....)
        newProMP.muY = ....  + LY.dot(desiredXmean[1] - basisMatrix.dot(....))
        newProMP.covMatY = .... - LY.dot(basisMatrix).dot(.....)
        newProMP.muZ = .... + LZ.dot(desiredXmean[2] - basisMatrix.dot(....))
        newProMP.covMatZ = s...  - LZ.dot(basisMatrix).dot(....)
        return newProMP




class MAPWeightLearner(): 

    def __init__(self, proMP, regularizationCoeff=10 ** -9, priorCovariance=10 ** -4, priorWeight=1):
        self.proMP = proMP
        self.priorCovariance = priorCovariance # sigma omega new
        self.priorWeight = priorWeight
        self.regularizationCoeff = regularizationCoeff


    def learnFromXDataTaskSapce(self, trajectoryList, timeList):  
        numTraj = len(trajectoryList)  # number of demos 
        print('numTraj=', numTraj)
        weightMatrix = np.zeros(... , ...) 
        for i in range(numTraj):
            trajectory = trajectoryList[i] 
            time = timeList[i] 
            trajectoryFlat = trajectory.transpose() 
            basisMatrix = self.proMP.basis.basisMultiDoF(time, 1)   
            temp = basisMatrix.transpose().dot(basisMatrix) + ....... 
            weightVector = np.linalg.solve(temp, ........)
            #print('weight vector size=', weightVector.shape)
            weightMatrix[i, :] = np.transpose(weightVector)  

        self.proMP.muX = np.mean(....... , axis=0) 

        sampleCov = np.cov(weightMatrix.transpose())
        self.proMP.covMatX = (numTraj * sampleCov + self.priorCovariance * np.eye(self.proMP.numWeights)) / (
                    numTraj + self.priorCovariance) 


    def learnFromYDataTaskSapce(self, trajectoryList, timeList):  
        numTraj = len(trajectoryList)  # number of demos 
        print('numTraj=', numTraj)
        weightMatrix = np.zeros((numTraj, self.proMP.numWeights)) 
        for i in range(numTraj):
            trajectory = trajectoryList[i]  
            time = timeList[i] 
            trajectoryFlat = trajectory.transpose() 
            basisMatrix = self.proMP.basis.basisMultiDoF(time, 1)   
            temp = basisMatrix.transpose().dot(basisMatrix) + ........
            weightVector = np.linalg.solve(temp, ...... )
            #print('weight vector size=', weightVector.shape)
            weightMatrix[i, :] = np.transpose(weightVector)  # from each demo learn a weight vector for a dim in task space

        self.proMP.muY = np.mean(....., axis=0) # 

        sampleCov = np.cov(weightMatrix.transpose())
        self.proMP.covMatY = (numTraj * sampleCov + self.priorCovariance * np.eye(self.proMP.numWeights)) / (
                    numTraj + self.priorCovariance) 


    def learnFromZDataTaskSapce(self, trajectoryList, timeList):  
        numTraj = len(trajectoryList)  # number of demos 
        print('numTraj=', numTraj)
        weightMatrix = np.zeros((numTraj, self.proMP.numWeights)) 
        for i in range(numTraj):
            trajectory = trajectoryList[i]  
            time = timeList[i] 
            trajectoryFlat = trajectory.transpose() 
            basisMatrix = self.proMP.basis.basisMultiDoF(time, 1)   # get the values of basis function at different time instant for multi dofs from single DoF 
            temp = basisMatrix.transpose().dot(basisMatrix) + ...........
            weightVector = np.linalg.solve(temp, ....... )
            #print('weight vector size=', weightVector.shape)
            weightMatrix[i, :] = np.transpose(weightVector)  

        self.proMP.muZ = np.mean(....... , axis=0) # mean of weights from prior , along rows

        sampleCov = np.cov(weightMatrix.transpose())
        self.proMP.covMatZ = (numTraj * sampleCov + self.priorCovariance * np.eye(self.proMP.numWeights)) / (
                    numTraj + self.priorCovariance) # cov of weight matrix




if __name__ == "__main__":

    phaseGenerator = phase.LinearPhaseGenerator()
    basisGenerator = basis.NormalizedRBFBasisGenerator(phaseGenerator, numBasis=5, duration=10, basisBandWidthFactor=5,
                                                       numBasisOutside=1)
    time = np.linspace(0, 1, 100)
    nDof = 1

    proMP = ProMP(basisGenerator, phaseGenerator, nDof) 

    ################################################################

