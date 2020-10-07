from fastai.vision.all import *
import matplotlib 

path = untar_data(URLs.MNIST_TINY)
mnist = DataBlock(blocks=(ImageBlock(cls=PILImageBW), CategoryBlock), 
                  get_items=get_image_files, 
                  splitter=GrandparentSplitter(),
                  get_y=parent_label)

dls = mnist.dataloaders(path)
dls.show_batch(max_n=9, figsize=(5,5))

matplotlib.pyplot.savefig('show_batch.png')

learn = cnn_learner(dls, resnet34, metrics=accuracy)
learn.fine_tune(3)
learn.recorder.plot_loss()

matplotlib.pyplot.savefig('loss.png')

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(5,5))

matplotlib.pyplot.savefig('confusion_matrix.png')
