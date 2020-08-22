'''
Viz package -- used for creation of visualizations in Pneumonia-CNN project
'''

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# def regroup_by_label(images,labels):
#     '''
#     Given an array of images a corresponding list of binary labels 0&1, returns 2 arrays of images, one for
#     label 0 and one for label 1
#     :param images:
#     array of images, of the form (dim1, dim2,
#     :param labels:
#     corresponding array of labels, of the same size as the images
#     :return:
#     '''


def overlap_images(list_of_images):
    '''

    :param list_of_images:
    list of images, expected form is grayscale with axis 0 as indexer
    images should be of same size
    :return:
    an overlap+spatial channel plot
    '''
    fig=plt.figure()
    fig.set_facecolor('gray')
    ax=fig.add_axes((0,0,1,1),label='o')
    #the label here is given only to totally forestall problems matplotlib can encounter with uniqueness of coordinates, since we're using very "standard" ones here
    ax.matshow(list_of_images[:,:,:].mean(axis=0),cmap='gray')
    #We take the list of images, dropping a possibly superfluous fourth coordinate, and find the mean along the assumed axis invidiaul images are assumed to be "strung along"
    ax.axis('off')

    #Relying on magic and wishes (trial and error) here for some of the locations,
    ax2=fig.add_axes((0.14,.98,.72,.3));
    #create a new ax object at the top of the original object-- slightly off to make things "fit right"

    # collapsing "down"
    ax2.patch.set_alpha(0)
    ax2.axis('off')
    ax2.plot(range(list_of_images[0,0,:].shape[0]),300*list_of_images[:,:,:].mean(axis=0).mean(axis=0).flatten(),c='w',lw=3);
    #The first mean here does as above, flattening each image "on top of" each other.  This results in a new array of shape (height,width,1).
    #Passing a second .mean(0) then averages along those rows, "collapsing the image down" and finding the mean along each vertical line of the image.
    #.flatten the coerces the right shape for matplotlib, the 300 is an example of an arcane number used for scaling (though more arbitrary)
    ax2.fill_between(range(list_of_images[0,0,:].shape[0]),300*list_of_images[:,:,:].mean(axis=0).mean(axis=0).flatten(),color='w',alpha=.7)
    #Same meat as the plt command

    # collapse right
    ax3=fig.add_axes((.82,-0.04,.3,1.08));
    ax3.patch.set_alpha(0)
    ax3.axis('off')
    ax3.plot(300*list_of_images[:,:,:].mean(axis=0).mean(axis=1).flatten(),range(list_of_images[0,0,:].shape[0]),c='w',lw=3);
    # This is essentially the same as what was described at length above, except notice the use of axis=1 in the second mean.
    # Here, the x and y coordinates are flipped to ensure the desired behavior occurs.
    ax3.fill_betweenx(range(list_of_images[0,0,:].shape[0]),0,300*list_of_images[:,:,:].mean(axis=0).mean(axis=1).flatten(),color='w',alpha=.7)
    #Note the use of "betweenx" instead of "between" here.
    leg=plt.legend(bbox_to_anchor=(1,1.2),handles=[mlines.Line2D([],[],lw=3,color='w',label='mean\nbrightness\nat level')],framealpha=0)
    plt.setp(leg.get_texts(), color='w')
    plt.close()

    return fig


def binary_sorting_viz(y_true,y_pred,target_phenomenon='Presence',steps=500,pause=100,cols=['blue','red']):

    '''

    :param y_true,y_pred:
    arrays or lists, the true and predicted labels for the data
    :param target_phenomenon:
    string, used for labelling
    :param steps:
    number of frames to use for movement
    :param pause:
    number of frames to wait for after movement, useful when using packages that don't handle ending commands
    well like IPython display widgets
    :param cols:
    list of colors, [0_cat color, 1_cat color]
    :return:
    a matplotlib animation object
    '''
    from celluloid import Camera
    from matplotlib.patches import Rectangle
    import matplotlib.patches as mpatches
    col0=cols[0]
    col1=cols[1]
    font_details = {'fontsize': 'large', 'ha': 'center', 'va': 'top'}

    init_pts_x = np.array([(-10 + 20 * s) + np.random.normal(scale=4) for s in y_true])
    init_pts_y = np.array([np.random.normal(scale=.2) for s in y_true])

    final_pts_x = np.array(
        [(s * (np.random.uniform(low=5, high=25)) + (1 - s) * (np.random.uniform(low=-25, high=-5)))[0] for s in
         y_pred])
    final_pts_y = np.array([((np.random.uniform(low=3, high=5))) for s in y_pred])

    fig, ax = plt.subplots()
    plt.ylim(-2, 5)
    plt.xlim(-28, 28)
    ax.axis('off')

    ax.text(-16, 2.8, f'No {target_phenomenon}', font_details)
    ax.text(14, 2.8, f'{target_phenomenon}', font_details)
    cam2 = Camera(fig)
    for t in np.linspace(0, 1, steps):
        ax.add_patch(Rectangle((-25, 3), 20, 2, fc=col0, alpha=.1))
        ax.add_patch(Rectangle((5, 3), 20, 2, fc=col1, alpha=.1))
        ax.text(-16, 2.8, f'No {target_phenomenon}\nPredicted', font_details)
        ax.text(14, 2.8, f'{target_phenomenon}\nPredicted', font_details)

        u = (init_pts_x * (1 - t) + final_pts_x * (t))
        v = (init_pts_y * (1 - t) + final_pts_y * (t))
        ax.legend(handles=[mpatches.Patch(color=col0, label=f'No {target_phenomenon}'),
                           mpatches.Patch(color=col1, label=f'Has {target_phenomenon}')], bbox_to_anchor=(0.5, 0),
                  loc='lower center', ncol=2)
        ax.scatter(u, v, color=[col1 if s == 1 else col0 for s in y_true], alpha=.4, edgecolors='black')
        cam2.snap()

    for r in range(pause):
        # adjust as linspace is adjusted
        ax.add_patch(Rectangle((-25, 3), 20, 2, fc=col0, alpha=.1))
        ax.add_patch(Rectangle((5, 3), 20, 2, fc=col1, alpha=.1))
        ax.text(-16, 2.8, f'No {target_phenomenon}\nPredicted', font_details)
        ax.text(14, 2.8, f'{target_phenomenon}\nPredicted', font_details)

        u = (final_pts_x)
        v = (final_pts_y)
        ax.legend(handles=[mpatches.Patch(color=col0, label=f'No {target_phenomenon}'),
                           mpatches.Patch(color=col1, label=f'Has {target_phenomenon}')], bbox_to_anchor=(0.5, 0),
                  loc='lower center', ncol=2)
        ax.scatter(u, v, color=[col1 if s == 1 else col0 for s in y_true], alpha=.4, edgecolors='black')
        cam2.snap()

        # Workaround to pause at end of video since matplotlib is awkward with that.
        # Very crude method done for expediency only
    plt.close()
    ani2 = cam2.animate()
    return ani2


def process_and_split_data(target_size=224,seed_val=60120,shuffling=False,batch_size=624,split_state=123):
    '''
    Performs tensorflow's flow_from_directory and sklearn's train_test_split on grayscale image data assumed to exist in the working
    directory under the folders "train","test", and "val" and then in binary-label subfolders.  Outputs images and labels arrays, and then test, train, and val splits for those. With a 20% cut at each.
    :param target_size:
    target_size param of flow_from_directory, assumed square
    :param seed_val:
    seed for flow_from_directory
    :param shuffling:
    value for flow_from_directory's shuffle param
    :param batch_size:
    batch_size param for flow_from_directory
    :param split_state:
    random_state for train_test_split
    :return:
    images,labels,X_train,y_train,X_test,y_test,X_val,y_val
    images&labels are all pulled for the batch, the rest are what they say on the tin.
    '''
    train_data = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        'train',
        target_size=(target_size, target_size), shuffle=shuffling, class_mode='binary',
        color_mode='grayscale',
        seed=seed_val,
        interpolation='nearest'
        ,
        batch_size=batch_size # using default interpolation scheme
    )

    train_images, train_labels = next(train_data)

    # Alphabetical order is used for classes, so normal is 0

    test_data = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        'test',
        target_size=(target_size, target_size), shuffle=shuffling, class_mode='binary',
        color_mode='grayscale',
        seed=seed_val,
        interpolation='nearest',
        batch_size=batch_size

    )

    test_images, test_labels = next(test_data)

    # Not sure what's going on with the batch sizes yet.
    val_data = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        'val',
        target_size=(target_size, target_size),
        shuffle=shuffling, class_mode='binary',
        color_mode='grayscale',
        seed=seed_val,
        interpolation='nearest',
        batch_size=batch_size

    )

    val_images, val_labels = next(val_data)

    images = np.concatenate((train_images, test_images, val_images))
    labels = np.concatenate((train_labels, test_labels, val_labels))

    X_model, X_test, y_model, y_test = train_test_split(images, labels, test_size=0.20, random_state=split_state)
    X_train, X_val, y_train, y_val = train_test_split(X_model, y_model, test_size=0.20, random_state=split_state)

    return images,labels,X_train,y_train,X_test,y_test,X_val,y_val

def pred_sorter(model, data, cutoff=.05):
    '''
    takes in a model and feature data and outputs values that will be pass the probability threshold, values that won't,
    and the mask used there.
    :param model:
    :param data:
    :param cutoff:
    :return:
    '''
    preds=model.predict(data)
    cat_preds=model.predict_classes(data).flatten()
    indexer=((preds<cutoff)|(preds>(1-cutoff))).flatten()
    nogo_preds=cat_preds[~indexer]
    good_preds=cat_preds[indexer]
    return good_preds,nogo_preds,indexer

# def describe_model(model,x,y):
#     #Like the .summary() method, but specialized
#
#     opt=str(model.optimizer).split()[0].split('.')[-1]
#     # This is brittle, it may fail for optimizers beyond tf's standard ones.
#
#     param_num=model.count_params()
#     eval_dict={k:v for k,v in zip(model.metrics_names,model.evaluate(x,y,verbose=0))}
#     #Create an evaluation dictionary from tf's .evaluate method by way of a list of keys & values
#     loss_fxn=model.loss
#
#     return opt,param_num,eval_dict

#Not used currently.

def model_perf(model,x,y):


