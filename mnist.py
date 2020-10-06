from fastai.vision.all import *

path = untar_data(URLs.PET)
block = DataBlock(blocks=(ImageBlock, CategoryBlock),
                  get_items=get_image_files,
                  get_y=parent_label,
                  splitter=GrandparentSplitter(),
                  item_tfms=Resize(224))
dls = block.dataloaders(path)
dls.show_batch()
savefig('show_batch.png')
