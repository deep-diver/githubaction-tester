from fastai.vision.all import *
import matplotlib 

path = untar_data(URLs.MNIST)
mnist = DataBlock(blocks=(ImageBlock(cls=PILImageBW), CategoryBlock), 
                  get_items=get_image_files, 
                  splitter=GrandparentSplitter(train_name='training', valid_name='testing'),
                  get_y=parent_label,
                  item_tfms=Resize(224))

dls = mnist.dataloaders(path)
dls.show_batch(max_n=9, figsize=(4,4))

matplotlib.pyplot.savefig('show_batch.png')
