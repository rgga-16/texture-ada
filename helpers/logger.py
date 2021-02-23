from args import args
from matplotlib import pyplot as plt

def log_args(path):
    f = open(path,"x")
    f.writelines([
        "Iters: {}\n".format(args.epochs),
        "Image size: {}\n".format(args.imsize),
        "Style Loss Weight: {:.4f}\n".format(args.style_weight),
        "Foreground MSELoss Weight: {:.4f}\n".format(args.foreground_weight),
    ])

def log_losses(losses,iterations,path,title='Loss History'):
    plt.plot(iterations,losses,label='Loss')
    plt.xlabel('No. iterations')

    plt.legend(loc='upper left')
    plt.title(title)
    plt.savefig(path)
    return