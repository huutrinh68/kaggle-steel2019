import torch 
from tqdm import tqdm 
import pandas as pd 
import numpy as np


def create_submission(model, device, configs, test_loader):
    coef        = configs["test"]["coef"]
    sample      = pd.read_csv(configs["data_path"]["sample_submission_csv"])
    batch_size  = configs["train"]["batch_size"]
    tta         = configs["test"]["tta"]
    test_pred   = np.zeros((len(sample), 1))
    model.eval()

    for _ in range(tta):
        with torch.no_grad():
            for i, data in tqdm(enumerate(test_loader)):
                images, _ = data
                images = images.to(device)
                pred = model(images)
                test_pred[i * batch_size:(i + 1) * batch_size] += pred.argmax(-1).detach().cpu().squeeze().numpy().reshape(-1,1)

    output = test_pred / tta

    
    for i, pred in tqdm(enumerate(output)):
        if pred < coef[0]:
            output[i] = 0
        elif pred >= coef[0] and pred < coef[1]:
            output[i] = 1
        elif pred >= coef[1] and pred < coef[2]:
            output[i] = 2
        elif pred >= coef[2] and pred < coef[3]:
            output[i] = 3
        else:
            output[i] = 4

    submission = pd.DataFrame({'id_code':sample.id_code.values,
                            'diagnosis':np.squeeze(output).astype(int)})
    submission.to_csv('submission.csv', index=False)
