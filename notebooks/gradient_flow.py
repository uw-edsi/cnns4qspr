import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import datetime
import numpy as np

def plot_grad_flow(epoch_number, batch_number, named_parameters, max_grad=False):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
#    ave_grads = []
#    layers = []
#    for n, p in named_parameters:
#        if(p.requires_grad) and ("bias" not in n):
#            layers.append(n)
#            ave_grads.append(p.grad.abs().mean())
#    layers = np.array(layers)
#    ave_grads = np.array(ave_grads)
#    print(layers)
#    plt.plot(ave_grads, alpha=0.3, color="b")
#    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
#    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
#    plt.xlim(xmin=0, xmax=len(ave_grads))
#    plt.xlabel("Layers")
#    plt.ylabel("average gradient")
#    plt.title("Gradient flow")
#    plt.grid(True)
#    plt.tight_layout()
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            if max_grad:
                max_grads.append(p.grad.abs().max())
    if max_grad:
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="b")
    plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.2, lw=1, color="g")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), [n[:18] for n in layers], rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow, epoch={}, batch={}".format(epoch_number, batch_number))
    plt.grid(True)
    if max_grad:
        plt.legend([Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="g", lw=4)],['max-gradient', 'ave-gradient'])
    else:
        plt.legend([Line2D([0], [0], color="g", lw=4)],['ave-gradient'])
    plt.tight_layout()
    dt = str(datetime.datetime.now())
    plt.savefig("_".join(["figs/", str(dt) ,"gradient_flow.png"]))

