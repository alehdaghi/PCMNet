import matplotlib
# from matplotlib import image as mpimg
# from matplotlib import interactive
# interactive(True)
# matplotlib.use('notebook')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torch.nn import functional as Fn
# plt.ion()

def on_pick(event):
    ind = event.ind
    # img = mpimg.imread(path[ind[0]])
    # img = Image.open(path[ind[0]])
    # img = img.resize((256,256))
    # img = np.asarray(img)
    XX = train_set[indices[ind[0]]][0]
    img = np.uint8(XX.permute(1, 2, 0).numpy() * 255)

    H,W,_ = img.shape
    mask = Fn.interpolate(M[ind[0], PART].unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear').squeeze()
    # img = np.uint8(XX[ind[0]].permute(1,2,0).numpy() * 255)
    heatmap_only, heatmap_on_image = apply_colormap_on_image(img, mask.cpu().numpy(), 'jet')

    ax[1].imshow(np.array(heatmap_on_image))
    plt.draw()
    print(ind[0], Y_C[ind[0]].item(), Y_Y[ind[0]].item())
    # artist = event.artist
    # xmouse, ymouse = event.mouseevent.xdata, event.mouseevent.ydata
    # print(artist.get_array().shape)
    # x, y = artist.get_xdata(), artist.get_ydata()
    # ind = event.ind
    # print('Artist picked:', event.artist)
    # print( '{} vertices picked'.format(len(ind)))
    # print( 'Pick between vertices {} and {}'.format(min(ind), max(ind)+1))
    # print( 'x, y of mouse: {:.2f},{:.2f}'.format(xmouse, ymouse))
    # print( 'Data point:', x[ind[0]], y[ind[0]])

# fig, ax = plt.subplots(1,2)
# # area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

# ax[0].scatter(emb[:,0], emb[:,1], c=Y_C, alpha=0.7, picker=True,cmap='Set1')

# fig.canvas.callbacks.connect('pick_event', on_pick)

# plt.show()

import matplotlib.cm as mpl_color_map
import copy
def apply_colormap_on_image(org_im, activation, colormap_name='jet'):
    """
        Apply heatmap on image
    Args:
        org_img (numpy arr): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))
    org_im = Image.fromarray(org_im)
    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image
