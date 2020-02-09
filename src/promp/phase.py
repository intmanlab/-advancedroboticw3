import numpy as np

class PhaseGenerator():

    def __init__(self):
        return

    def phase(self, time):

        # Base class...
        return



class LinearPhaseGenerator(PhaseGenerator): 

    def __init__(self, phaseVelocity = 1.0):

        PhaseGenerator.__init__(self)
        self.phaseVelocity = phaseVelocity


    def phase(self, time):

        phase = np.array(time) * self.phaseVelocity  # phase = z = z_dot = z/t --> z = z_dot * time 

        return phase



if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np

    phaseGenerator = LinearPhaseGenerator()
    time = np.linspace(0,1, 100)
    phase =  phaseGenerator.phase(time)

    plt.figure()
    plt.plot(time, phase)
    plt.hold(True)

    phaseGenerator.tau = 2
    phase = phaseGenerator.phase(time)
    plt.plot(time, phase)

    phaseGenerator.tau = 0.5
    phase = phaseGenerator.phase(time)
    plt.plot(time, phase)

    plt.show()


    print('PhaseGeneration Done')

