import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common import *
from dataset import *



def valid_augment(image, mask, infor):
    return image, mask, infor



def train_augment(image, mask, infor):
    u=np.random.choice(3)
    if u==0:
        pass
    elif u==1:
        image_mask = do_random_crop_rescale(image, mask, 1600-(256-224), 224)

    return image, mask, infor



def run_train():
    out_dir = 'efficientnet-b5'

    # hyperparams
    batch_size  = 20
    num_workers = 10


    ## setup --------------------------------------------------------------------------------------
    for f in ['checkpoint', 'train', 'valid', 'backup']: os.makedirs(out_dir+'/'+f, exist_ok=True)
    # backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')

    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## dataset ------------------------------------------------------------------------------------
    log.write('** dataset setting **\n')

    train_dataset = SteelDataset(
        mode    = 'train',
        csv     = ['train.csv',],
        split   = ['train0_11568.npy'],
        augment = train_augment,
    )
    train_loader  = DataLoader(
        train_dataset,
        sampler     = RandomSampler(train_dataset),
        batch_size  = batch_size,
        drop_last   = True,
        num_workers = num_workers,
        pin_memory  = True,
        collate_fn  = null_collate
    )

    valid_dataset = SteelDataset(
        mode    = 'train',
        csv     = ['train.csv',],
        split   = ['valid0_1000.npy',],
        augment = valid_augment,
    )
    valid_loader = DataLoader(
        valid_dataset,
        sampler    = SequentialSampler(valid_dataset),
        #sampler     = RandomSampler(valid_dataset),
        batch_size  = batch_size,
        drop_last   = False,
        num_workers = num_workers,
        pin_memory  = True,
        collate_fn  = null_collate
    )
    assert(len(train_dataset)>=batch_size)
    log.write('batch_size = %d\n'%(batch_size))
    log.write('train_dataset : \n%s\n'%(train_dataset))
    log.write('valid_dataset : \n%s\n'%(valid_dataset))
    log.write('\n')


# main ############################################################################################
if __name__ == "__main__":
    print( '%s calling main function ...' % os.path.basename(__file__))

    run_train()