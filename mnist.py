import subprocess
import argparse
import wandb
from fastai.vision.all import *
from fastai.callback.wandb import *
import matplotlib 

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

def set_wandb(key, sha):
	wandb.login(key=key)
	ro = wandb.init(project='test.project')
	
	target_file = open('template/wandb.md')
	new_file_content = ""
	
	for line in target_file:
		stripped_line = line.strip()
		new_line = stripped_line.replace("t_wandb_link", ro.get_url())
		new_file_content += new_line +"\n"
	target_file.close()
	
	target_file = open("generated_template/wandb.md", "w")
	target_file.write(new_file_content)
	target_file.close()

	subprocess.run(f"cml-send-comment --commit-sha {sha} generated_template/wandb.md", shell=True)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Please specify wandb key if you want')
	parser.add_argument('--wandb_key', type=str) 
	parser.add_argument('--sha', type=str)
	args = parser.parse_args()

	if args.wandb_key:
		set_wandb(args.wandb_key, args.sha)

	PATH = Path('data/')
	train = pd.read_csv(f'data/train.csv')

	db = DataBlock(blocks=(ImageBlock(cls=PILImageBW), CategoryBlock),
          	get_items=get_items,
          	get_x=get_x,
          	splitter=RandomSplitter(seed=42),
          	get_y=get_y,
		item_tfms=Resize(224))

	dls = DataLoaders.from_dblock(db, df=train, source=PATH, bs=32)
	dls.show_batch(max_n=5, ncols=5, nrows=1, figsize=(6,2))

	matplotlib.pyplot.savefig('show_batch.png')
	matplotlib.pyplot.clf()

	learn = cnn_learner(dls, resnet34, metrics=[accuracy, error_rate])
	if args.wandb_key:
		learn.fine_tune(5, cbs=WandbCallback())
	else:
		learn.fine_tune(5)

	matplotlib.pyplot.figure(figsize=(6,3))
	learn.recorder.plot_loss()
	matplotlib.pyplot.savefig('loss.png')
	matplotlib.pyplot.clf()

	interp = ClassificationInterpretation.from_learner(learn)
	interp.plot_confusion_matrix(figsize=(5,5))
	matplotlib.pyplot.savefig('confusion_matrix.png')
	matplotlib.pyplot.clf()
