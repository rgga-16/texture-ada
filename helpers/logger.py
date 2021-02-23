from args import args

def log_args(path):
    f = open(path,"x")
    f.writelines([
        "Iters: {}\n".format(args.epochs),
        "Image size: {}\n".format(args.imsize),
        "Style Loss Weight: {:.4f}\n".format(args.style_weight),
        "Foreground MSELoss Weight: {:.4f}\n".format(args.foreground_weight),
    ])
