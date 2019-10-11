import os
import numpy as np
import pandas as pd
import glob
import random

# https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/107053#latest-617549
DUPLICATE=np.array([
    'train_images/6eb8690cd.jpg', 'train_images/a67df9196.jpg',
    'train_images/24e125a16.jpg', 'train_images/4a80680e5.jpg',
    'train_images/a335fc5cc.jpg', 'train_images/fb352c185.jpg',
    'train_images/c35fa49e2.jpg', 'train_images/e4da37c1e.jpg',
    'train_images/877d319fd.jpg', 'train_images/e6042b9a7.jpg',
    'train_images/618f0ff16.jpg', 'train_images/ace59105f.jpg',
    'train_images/ae35b6067.jpg', 'train_images/fdb5ae9d4.jpg',
    'train_images/3de8f5d88.jpg', 'train_images/a5aa4829b.jpg',
    'train_images/3bd0fd84d.jpg', 'train_images/b719010ac.jpg',
    'train_images/24fce7ae0.jpg', 'train_images/edf12f5f1.jpg',
    'train_images/49e374bd3.jpg', 'train_images/6099f39dc.jpg',
    'train_images/9b2ed195e.jpg', 'train_images/c30ecf35c.jpg',
    'train_images/3a7f1857b.jpg', 'train_images/c37633c03.jpg',
    'train_images/8c2a5c8f7.jpg', 'train_images/abedd15e2.jpg',
    'train_images/b46dafae2.jpg', 'train_images/ce5f0cec3.jpg',
    'train_images/5b1c96f09.jpg', 'train_images/e054a983d.jpg',
    'train_images/3088a6a0d.jpg', 'train_images/7f3181e44.jpg',
    'train_images/dc0c6c0de.jpg', 'train_images/e4d9efbaa.jpg',
    'train_images/488c35cf9.jpg', 'train_images/845935465.jpg',
    'train_images/3b168b16e.jpg', 'train_images/c6af2acac.jpg',
    'train_images/05bc27672.jpg', 'train_images/dfefd11c4.jpg',
    'train_images/048d14d3f.jpg', 'train_images/7c8a469a4.jpg',
    'train_images/a1a0111dd.jpg', 'train_images/b30a3e3b6.jpg',
    'train_images/d8be02bfa.jpg', 'train_images/e45010a6a.jpg',
    'train_images/caf49d870.jpg', 'train_images/ef5c1b08e.jpg',
    'train_images/63c219c6f.jpg', 'train_images/b1096a78f.jpg',
    'train_images/76096b17b.jpg', 'train_images/d490180a3.jpg',
    'train_images/bd0e26062.jpg', 'train_images/e7d7c87e2.jpg',
    'train_images/600a81590.jpg', 'train_images/eb5aec756.jpg',
    'train_images/ad5a2ea44.jpg', 'train_images/e9fa75516.jpg',
    'train_images/6afa917f2.jpg', 'train_images/9fb53a74b.jpg',
    'train_images/59931eb56.jpg', 'train_images/e7ced5b76.jpg',
    'train_images/0bfe252d0.jpg', 'train_images/b4d0843ed.jpg',
    'train_images/67fc6eeb8.jpg', 'train_images/c04aa9618.jpg',
    'train_images/741a5c461.jpg', 'train_images/dae3c563a.jpg',
    'train_images/78416c3d0.jpg', 'train_images/e34f68168.jpg',
    'train_images/0d258e4ae.jpg', 'train_images/72322fc23.jpg',
    'train_images/0aafd7471.jpg', 'train_images/461f83c57.jpg',
    'train_images/38a1d7aab.jpg', 'train_images/8866a93f6.jpg',
    'train_images/7c5b834b7.jpg', 'train_images/dea514023.jpg',
    'train_images/32854e5bf.jpg', 'train_images/530227cd2.jpg',
    'train_images/1b7d7eec6.jpg', 'train_images/f801dd10b.jpg',
    'train_images/46ace1c15.jpg', 'train_images/876e74fd6.jpg',
    'train_images/578b43574.jpg', 'train_images/9c5884cdd.jpg',
]).reshape(-1,2).tolist()


def make_mask(idx, df):
    '''
    Given a row index, return image_id and mask (256, 1600, 4) from the dataframe df
    Parameters
    ----------
    idx: int
        Row index
    df: dataframe
        List of index of image

    Returns
    ----------
    fname: int
        Id of image
    mask: triple
        Mask corresponding to image
    '''

    fname  = df.iloc[idx].name
    labels = df.iloc[idx][:4]
    masks  = np.zeros((256, 1600, 4), dtype=np.float32)

    for i, label in enumerate(labels.values):
        if label is not np.nan:
            label     = label.split(" ")
            positions = map(int, label[0::2])
            length    = map(int, label[1::1])
            mask = np.zeros(256*1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos:(pos+le)] = 1
            masks[:, :, i] = mask.reshape(256, 1600, order='F')

    return fname, masks


def run_make_train_split():

    image_file =  glob.glob('../data/train_images/*.jpg')
    image_file = ['train_images/'+i.split('/')[-1] for i in image_file]
    print(len(image_file)) #1256
    #print(image_file[:10])

    #without duplicate
    duplicate = np.array(DUPLICATE).reshape(-1).tolist() #88
    non_duplicate = list(set(image_file)-set(duplicate)) #12480
    random.shuffle(non_duplicate)


    #12480
    num_fold  = 2
    num_valid = 500
    for n in range(num_fold):
        valid = non_duplicate[n*num_valid:(n+1)*num_valid]
        train = list(set(non_duplicate)-set(valid))
        assert len(list(set(train).intersection(valid))) == 0, "the list is non empty"

        os.makedirs('../data/split/', exist_ok=True)
        np.save('../data/split/train_a%d_%d.npy'%(n, len(train)), train)
        np.save('../data/split/valid_a%d_%d.npy'%(n, len(valid)), valid)

def run_make_test_split():

    df =  pd.read_csv('../data/sample_submission.csv')
    df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
    uid = df['ImageId'].unique().tolist()

    test = ['test_images/'+i for i in uid]
    os.makedirs('../data/split/', exist_ok=True)
    np.save('../data/split/test_%d.npy'%len(test),test)

    




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_make_train_split()
    run_make_test_split()
