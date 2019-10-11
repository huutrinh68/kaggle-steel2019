from glob import glob
import cv2
from tqdm import tqdm
import pickle
import os
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor


def average_hash(img):
    img = cv2.resize(img, (8, 8 * 6))
    mean = img[img > 10].mean()
    h = (img > mean).tobytes()
    return h


train_image_fns = sorted(glob('../data/train_images/*.jpg'))
test_image_fns = sorted(glob('../data/test_images/*.jpg'))

fns = train_image_fns + test_image_fns


def compute_hash(fn):
    img = cv2.imread(fn, 0)
    h = average_hash(img)
    return h, fn


cache_fn = 'hash_cache.p'
if not os.path.exists(cache_fn):
    with ThreadPoolExecutor() as e:
        hash_fn = list(tqdm(e.map(compute_hash, fns), total=len(fns)))
    with open(cache_fn, 'wb') as f:
        pickle.dump(hash_fn, f)
else:
    with open(cache_fn, 'rb') as f:
        hash_fn = pickle.load(f)


hashes = defaultdict(list)
for h, fn in hash_fn:
    hashes[h].append(fn)


duplicates = []
plot = False
for k in tqdm(sorted(hashes)[::-1]):
    duplicate_fns = hashes[k]
    if len(duplicate_fns) >= 2:
        vis = []
        is_duplicate = False
        ref_img = cv2.imread(duplicate_fns[0], 0)
        vis.append(ref_img)
        for fn in duplicate_fns[1:]:
            img = cv2.imread(fn, 0)
            vis.append(img)
            eq = img == ref_img
            eq[0: 15, 195: 640] = True
            if np.all(eq):
                is_duplicate = True
                duplicates.append(duplicate_fns)
                break

        if is_duplicate:
            if plot:
                vis = np.vstack(vis)
                plt.imshow(vis)
                plt.show()

print("%d of %d are duplicates" % (len(duplicates), len(fns)))
duplicates = [','.join(d) + '\n' for d in duplicates]
with open('duplicates.csv', 'w') as f:
    f.writelines(duplicates)