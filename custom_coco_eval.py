import argparse

import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms

from confusion_matrix import ConfusionMatrix
from retinanet.dataloader import CocoDataset, Resizer, Normalizer

import seaborn as sn


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--coco_path', help='Path to COCO directory')

    parser = parser.parse_args(args)

    dataset_val = CocoDataset(parser.coco_path, set_name='validation',
                              transform=transforms.Compose([Normalizer(), Resizer()]))

    coco_true = dataset_val.coco
    coco_pred = coco_true.loadRes('validation_bbox_results.json')
    num_of_classes = 13

    conf_mat = ConfusionMatrix(num_classes=num_of_classes, CONF_THRESHOLD=0.3, IOU_THRESHOLD=0.5)

    for k, v in coco_true.imgToAnns.items():
        elems_true = np.ndarray(shape=(len(v), 5))
        elems_pred = np.ndarray(shape=(len(coco_pred.imgToAnns[k]), 6))

        for i, a in enumerate(v):
            elems_true[i][0] = a['category_id']
            elems_true[i][1] = a['bbox'][0]
            elems_true[i][2] = a['bbox'][1]
            elems_true[i][3] = a['bbox'][0] + a['bbox'][2]
            elems_true[i][4] = a['bbox'][0] + a['bbox'][3]

        for i, a in enumerate(coco_pred.imgToAnns[k]):
            elems_pred[i][0] = a['bbox'][0]
            elems_pred[i][1] = a['bbox'][1]
            elems_pred[i][2] = a['bbox'][0] + a['bbox'][2]
            elems_pred[i][3] = a['bbox'][0] + a['bbox'][3]
            elems_pred[i][4] = a['score']
            elems_pred[i][5] = a['category_id']

        conf_mat.process_batch(elems_pred, elems_true)

    print("Finished!")
    conf_mat.print_matrix()

    cats = []
    cm = conf_mat.return_matrix()[:num_of_classes, :num_of_classes]

    indicesToRemove = []

    for i in range(0, num_of_classes):
        if i not in coco_true.cats:
            indicesToRemove.append(i)
        else:
            cats.append(coco_true.cats[i]['name'])

    cm = np.delete(cm, indicesToRemove, 0)
    cm = np.delete(cm, indicesToRemove, 1)


    plt.figure(figsize=(20, 20))
    ax = sn.heatmap(cm / np.sum(cm), annot=True,
                     fmt='.2%', cmap='Blues')

    ax.set_title('BIAI confusion matrix\n\n')
    ax.set_xlabel('\nActual')
    ax.set_ylabel('Predicted')

    ax.xaxis.set_ticklabels(cats)
    ax.yaxis.set_ticklabels(cats)

    plt.savefig("confusion_matrix.png")


if __name__ == "__main__":
    main()
