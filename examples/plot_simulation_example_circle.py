import matplotlib.pyplot as plt
import numpy as np
from layers import motion_sim

def plot_im(im,title=None,*args,**kwargs):
    plt.imshow(im,cmap='gray',*args,**kwargs),plt.axis('off')
    if title:
        plt.title(title)


input_size = [100,100,100]
image = np.zeros(input_size)
xx, yy = np.mgrid[:input_size[0], :input_size[1]]
circle = ((xx - input_size[0]/2)**2 + (yy - input_size[1]/2)**2) < 800
image[5:-5] = circle

ms_layer = motion_sim.MotionSimLayer(corrupt_pct_range=[30,30],
                                     corruption_scheme='piecewise_constant',n_seg=8)

output = ms_layer.layer_op(image)

translations = np.reshape(ms_layer.translations.T, image.shape + (3,))
translations_sq = translations**2
trans_magnitude = np.sqrt(translations_sq.sum(axis=-1))

plt.figure()
plt.subplot(1,3,1)
plot_im(trans_magnitude[:,10,:],interpolation='none',title='motion magnitude\n(Fourier domain)')
plt.subplot(1,3,2)
plot_im(image[10],title='original')
plt.subplot(1,3,3)
plot_im(output[10],title='motion simulation')
plt.show()