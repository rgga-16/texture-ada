from matplotlib import pyplot as plt 
import numpy as np


def display_losses(losses,iterations,title='Loss History'):
    plt.plot(iterations,losses,label='Loss')
    plt.xlabel('No. iterations')

    plt.legend(loc='upper left')
    plt.title(title)
    plt.show()


if __name__ == '__main__':

    niters = 5
    losses = np.random.rand(niters)
    iterations = range(niters)
    display_losses(losses,iterations,'Losses of Style Transfer')
