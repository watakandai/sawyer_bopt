import time
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import os, os.path
import dill

############################################################
############################################################
# REQUIRES MODIFICATION ON WHERE TO LOAD THE FILE FROM
# AND THE NAME OF THOSE FILES
############################################################
############################################################


def anim_acquisition(i_eval, evaluate_type, title):
    exp_name = exp_names[i_eval]
    DIR = './bopt_data/bopt_3d/%s/ex1'%(exp_name)

    setting_filename = 'bopt_38.pkl'
    files = os.listdir(DIR)
    n = len(files)
    print('%i files'%(n))

    filename = os.path.join(DIR, setting_filename)
    print('setting from file %s ...'%(filename))
    with open(filename, 'rb') as f:
        bopt = dill.loads(f.read())
    print(bopt.Y)

    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.grid(True)
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.set_title(title)
    bounds = bopt.acquisition.space.get_bounds()
    ax1.set_xlim(bounds[0][0],bounds[0][1]) 
    ax1.set_ylim(bounds[1][0],bounds[1][1])
    ax1.set_zlim(bounds[2][0],bounds[2][1])
    frames = []
    num_sample = 50

    for i_trial in range(1, n):
        filename = os.path.join(DIR,'bopt_%i.pkl'%(i_trial))
        print('opening file %s ...'%(filename))
        with open(filename, 'rb') as f:
            bopt = dill.loads(f.read())

        X1,X2,X3, m,v,acqu, next_sample = bopt.calculate_for_3d_plot(num_sample)
        i_frames = []
        for i_z in range(num_sample):
            frame = bopt.plot_3d_frame(ax1, i_trial, i_z, evaluate_type, X1,X2,X3, m,v,acqu, next_sample)
            i_frames += frame
        frames.append(i_frames)

    anim = animation.ArtistAnimation(fig, frames, interval=100.0) 
    #anim.save('animation/%s.gif'%(exp_name), writer='imagemagick')
    #print('gif created')
    anim.save('animation/%s.mp4'%(exp_name+'_'+title), writer='ffmpeg')
    print('mp4 created')
    #s = anim.to_jshtml()
    #with open('animation/%s.html'%(exp_name), 'w') as f:
    #    f.write(s)
    #print('html created')
    #plt.show() 
    print('Animation Done!')


if __name__ == '__main__':
    """
    CHOOSE, tennis/redball & shallow/deep
    for dir, and anim
    """
    evaluate_types = [2,2] # 'Acquisition Function', 'Standard Deviation'
    titles = ['std']
    exp_names = ['redball_deep', 'tennis_shallow'] # 'redball_shallow', 'tennis_deep', 'tennis_shallow'
    for i_eval, evaluate_type in enumerate(evaluate_types):
        for title in titles:
            anim_acquisition(i_eval, evaluate_type, title)