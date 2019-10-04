import numpy as np

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
    fname = df.iloc[idx].name
    labels = df.iloc[idx][:4]
    masks = np.zeros((256, 1600, 4), dtype=np.float32)

    for i, label in enumerate(labels.values):
        if label is not np.nan:
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::1])
            mask = np.zeros(255*1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos:(pos+le)] = 1
            masks[:, :, i] = mask.reshape(255, 1600, order='F')

    return fname, masks