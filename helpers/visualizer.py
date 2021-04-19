import torch
from matplotlib import pyplot as plt 
import numpy as np

from skimage import io 
from PIL import Image

import os
import plotly.graph_objects as go

def display_losses(losses,iterations,title='Loss History'):
    plt.plot(iterations,losses,label='Loss')
    plt.xlabel('No. iterations')

    plt.legend(loc='upper left')
    plt.title(title)
    plt.show()

def display_color_distrib(filename:str,min_val=0,max_val=255,show:bool=True,save:bool=True, save_path='./color_distrib.png'):

    image = io.imread(filename)

    _ = plt.hist(np.array([c for c in image.ravel() if c >= min_val and c <= max_val]), bins=256,color='orange',alpha=0.3)

    _ = plt.hist(np.array([r for r in image[:,:,0].ravel() if r >= min_val and r <= max_val]), bins=256,color='red',alpha=0.5)
    _ = plt.hist(np.array([g for g in image[:,:,1].ravel() if g >= min_val and g <= max_val]), bins=256,color='green',alpha=0.5)
    _ = plt.hist(np.array([b for b in image[:,:,2].ravel() if b >= min_val and b <= max_val]), bins=256,color='blue',alpha=0.5)
    _ = plt.hist(np.array([a for a in image[:,:,3].ravel() if a >= min_val and a <= max_val]), bins=256,color='grey',alpha=1.0)
 
    _ = plt.xlabel('Intensity Value')
    _ = plt.ylabel('Count')
    # _ = plt.legend(['Red_Channel', 'Green_Channel', 'Blue_Channel'])
    _ = plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel','Alpha_Channel'])


    if show:
        plt.show()

    if save:
        plt.savefig(save_path)
    
    plt.clf()

    return

def visualize_rotate(data):
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames=[]

    def rotate_z(x, y, z, theta):
        w = x+1j*y
        return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))
    fig = go.Figure(data=data,
                    layout=go.Layout(
                        updatemenus=[dict(type='buttons',
                                    showactive=False,
                                    y=1,
                                    x=0.8,
                                    xanchor='left',
                                    yanchor='bottom',
                                    pad=dict(t=45, r=10),
                                    buttons=[dict(label='Play',
                                                    method='animate',
                                                    args=[None, dict(frame=dict(duration=50, redraw=True),
                                                                    transition=dict(duration=0),
                                                                    fromcurrent=True,
                                                                    mode='immediate'
                                                                    )]
                                                    )
                                            ]
                                    )
                                ]
                    ),
                    frames=frames
            )

    return fig

def display_pointcloud(pointcloud):
    xs,ys,zs = np.array(pointcloud.squeeze().cpu()).T
    data=[go.Scatter3d(x=xs, y=ys, z=zs,
                                   mode='markers')]
    # fig = visualize_rotate(data)
    fig = go.Figure(data=data)
    fig.update_traces(marker=dict(size=2,
                      line=dict(width=2,
                      color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()

if __name__ == '__main__':

    
    # images = [
    # 'uv_map_backseat_chair-2_tiled.png',
    # 'uv_map_left_arm_chair-3_tiled.png',
    # 'uv_map_right_arm_chair-3_tiled.png',
    # ]

    images = [
    'chair-2_tiled.png',
    'chair-3_tiled.png',
    ]
    
    root_ = './inputs/style_images/tiled'
    # root_ = './outputs/output_images/Pyramid2D_with_instnorm/armchair_sofa/[3-11-21 15-00]'
    # root_ = './outputs/gram vs cov'



    for root,dirs,files in os.walk(root_):
        
        for file in files:
            if file in images:
                file_path = os.path.join(root_,file)
                save_file = "{}_histogram_notips.png".format(file[:-4])
                save_path = os.path.join(root_,save_file)
                display_color_distrib(file_path,10,245,show=False,save=True,save_path=save_path)
