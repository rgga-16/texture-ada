import args as args_
args = args_.parse_arguments()
from matplotlib import pyplot as plt

def log_args(path,**kwargs):
    f = open(path,"x")
    f.writelines([
        "Configurations\n\n",
        "="*10,"\n",
        "Iters: {}\n".format(args.epochs),
        "Style image size: {}\n".format(args.style_size),
        "UV Map train sizes: {}\n".format(args.uv_train_sizes),
        "UV Map test sizes: {}\n".format(args.uv_test_sizes),
        "Output image size: {}\n".format(args.output_size),
        "Learning rate: {}\n".format(args.lr),
        "="*10,"\n",
        "Style Loss Weight: {:.4f}\n".format(args.style_weight),
        "="*10,"\n",
        "Input UV Maps Directory: {}\n".format(args.uv_dir),
        "Style Images Directory: {}\n".format(args.style_dir),
        "UV-Style Pairs: {}\n".format(args.uv_style_pairs),
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