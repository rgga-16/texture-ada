from args import args
from matplotlib import pyplot as plt

def log_args(path,**kwargs):
    f = open(path,"x")
    f.writelines([
        "Configurations\n\n",
        "="*10,"\n",
        "Iters: {}\n".format(args.epochs),
        "Image size: {}\n".format(args.imsize),
        "Learning rate: {}\n".format(args.lr),
        "="*10,"\n",
        "Style Loss Weight: {:.4f}\n".format(args.style_weight),
        "Foreground MSELoss Weight: {:.4f}\n".format(args.foreground_weight),
        "="*10,"\n",
        "Input Texture Maps Directory: {}\n".format(args.content_dir),
        "Style Images Directory: {}\n".format(args.style_dir),
        "Outputs Directory: {}\n".format(args.output_dir),
        "="*10,"\n\n",
    ])

    for k,v in kwargs.items():
        f.write("{}: {}\n".format(k,v))

    f.close()


def log_losses(losses,iterations,path,title='Loss History'):
    plt.plot(iterations,losses,label='Loss')
    plt.xlabel('No. iterations')

    plt.legend(loc='upper left')
    plt.title(title)
    plt.savefig(path)
    return