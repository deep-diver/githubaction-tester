from fastai.vision.all import *
import matplotlib 

matplotlib.rcParams['figure.figsize'] = (3.0, 3.0)

path = untar_data(URLs.MNIST_TINY)
mnist = DataBlock(blocks=(ImageBlock(cls=PILImageBW), CategoryBlock), 
                  get_items=get_image_files, 
                  splitter=GrandparentSplitter(),
                  get_y=parent_label)

dls = mnist.dataloaders(path)
dls.show_batch(max_n=9, figsize=(5,5))

matplotlib.pyplot.savefig('show_batch.png')
matplotlib.pyplot.clf()

learn = cnn_learner(dls, resnet34, metrics=accuracy)
learn.fine_tune(3)
learn.recorder.plot_loss()

matplotlib.pyplot.savefig('loss.png')
matplotlib.pyplot.clf()

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(5,5))

matplotlib.pyplot.savefig('confusion_matrix.png')
matplotlib.pyplot.clf()
