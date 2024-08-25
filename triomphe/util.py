import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors

def plot_one(ax, i, task, train_or_test, input_or_output):
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    
    input_matrix = task[train_or_test][i][input_or_output]
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True,which='both',color='lightgrey', linewidth=0.5)    
    ax.set_yticks([x-0.5 for x in range(1+len(input_matrix))])
    ax.set_xticks([x-0.5 for x in range(1+len(input_matrix[0]))])     
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(train_or_test + ' '+input_or_output)
    

def plot_task(task, outprefix):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """    
    num_train = len(task['train'])
    fig, axs = plt.subplots(2, num_train, figsize=(3*num_train,3*2))
    for i in range(num_train):     
        plot_one(axs[0,i],i,task,'train','input')
        plot_one(axs[1,i],i,task,'train','output')        
    plt.tight_layout()
    outfile = outprefix + "_train.png"
    plt.savefig(outfile)
    print("Wrote out to {}.".format(outfile))        
    
    num_test = len(task['test'])
    for i in range(num_test):
        predictions = [p for p in task['test'][i]['predictions']]
        height = len(predictions[0])
        padding = [np.zeros((height, 2), dtype=np.uint8) for _ in range(len(predictions)-1)]
        mats_with_padding = [predictions[j//2] if (j % 2 == 0) else padding[(j-1)//2] \
            for j in range(len(predictions)+len(padding))]
        task['test'][i]['prediction'] = np.hstack(mats_with_padding)

    fig, axs = plt.subplots(3, num_test, figsize=(3*num_test,3*2))
    if num_test==1: 
        plot_one(axs[0],0,task,'test','input')
        plot_one(axs[1],0,task,'test','prediction')
        plot_one(axs[2],0,task,'test','output')     
    else:
        for i in range(num_test):      
            plot_one(axs[0,i],i,task,'test','input')
            plot_one(axs[1,i],i,task,'test','prediction')
            plot_one(axs[2,i],i,task,'test','output')  
    plt.tight_layout()
    outfile = outprefix + "_test.png"
    plt.savefig(outfile)
    print("Wrote out to {}.".format(outfile))
