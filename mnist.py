from fastai.vision.all import *
import matplotlib 

train = pd.read_csv(f'data/train.csv')

def get_items(path):
    train_df = pd.read_csv(f'{PATH}/train.csv')
    return train_df.to_numpy().tolist()

def get_x(item):
    if len(item) == 785:
        x = np.array(item[1:]).reshape(28, 28).astype(np.uint8)
    else:
        x = np.array(item).reshape(28, 28).astype(np.uint8)
    return x

def get_y(item):
    return item[0]

db = DataBlock(blocks=(ImageBlock(cls=PILImageBW), CategoryBlock),
          get_items=get_items,
          get_x=get_x,
          splitter=RandomSplitter(seed=42),
          get_y=get_y)

dls = DataLoaders.from_dblock(db, df=train, source=PATH, bs=128)
dls.show_batch(max_n=5, ncols=5, nrows=1, figsize=(6,2))

matplotlib.pyplot.savefig('show_batch.png')
matplotlib.pyplot.clf()

learn = cnn_learner(dls, resnet34, metrics=[accuracy, error_rate])
learn.fine_tune(5)

matplotlib.pyplot.figure(figsize=(6,3))
learn.recorder.plot_loss()
matplotlib.pyplot.savefig('loss.png')
matplotlib.pyplot.clf()

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(5,5))
matplotlib.pyplot.savefig('confusion_matrix.png')
matplotlib.pyplot.clf()
