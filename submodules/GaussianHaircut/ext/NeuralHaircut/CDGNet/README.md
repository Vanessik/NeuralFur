This repo is a PyTorch implementation of our paper CDGNet: Class Distribution Guided Network for Human Parsing accepted by CVPR2022(https://openaccess.thecvf.com/content/CVPR2022/html/Liu_CDGNet_Class_Distribution_Guided_Network_for_Human_Parsing_CVPR_2022_paper.html). We accumulate the original human parsing Ground Truth in the horizontal and vertical directions to obtain the class distribution lables that can guide the network to exploit the intrinsic distribution rule of each class. The generated labels can act as additional supervision signal to improve the parsing performance.

Requirements
Pytorch 1.9.0

Python 3.7

Implementation

Dataset
Please download LIP dataset and make them follow this structure:
'''
|-- LIP
    |-- images_labels
        |-- train_images
        |-- train_segmentations
        |-- val_images
        |-- val_segmentations
        |-- train_id.txt
        |-- val_id.txt
'''
Please download imagenet pretrained resent-101 from [baidu drive](https://pan.baidu.com/s/1NoxI_JetjSVa7uqgVSKdPw) or [Google drive](https://drive.google.com/open?id=1rzLU-wK6rEorCNJfwrmIu5hY2wRMyKTK), and put it into dataset folder.

### Training and Evaluation
```bash
./run.sh
```
Please download the trained model for LIP dataset from [baidu drive](https://pan.baidu.com/s/1WyifPOOE0SqIzCje-d1kzA?pwd=81la) and put it into snapshots folder.

./run_evaluate.sh for single scale evaluation or ./run_evaluate_multiScale.sh for multiple scale evaluation.
``` 
The parsing result of the provided 'LIP_epoch_149.pth' is 60.30 on LIP dataset.

Citation:

If you find our work useful for your research, please cite:

@InProceedings{Liu_2022_CVPR,
    author    = {Liu, Kunliang and Choi, Ouk and Wang, Jianming and Hwang, Wonjun},
    title     = {CDGNet: Class Distribution Guided Network for Human Parsing},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {4473-4482}
}

Acknowledgement:

  We acknowledge Ziwei Zhang and Tao Ruan for sharing their codes.
