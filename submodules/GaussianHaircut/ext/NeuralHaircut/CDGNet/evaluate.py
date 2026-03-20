import argparse
import numpy as np
import torch
torch.multiprocessing.set_start_method("spawn", force=True)
from torch.utils import data
from networks.CDGNet import Res_Deeplab
from dataset.datasets import LIPDataSet
import os
import torchvision.transforms as transforms
from utils.miou import compute_mean_ioU
from copy import deepcopy

from PIL import Image as PILImage

DATA_DIRECTORY = '/ssd1/liuting14/Dataset/LIP/'
DATA_LIST_PATH = './dataset/list/lip/valList.txt'
IGNORE_LABEL = 255
NUM_CLASSES = 20
SNAPSHOT_DIR = './snapshots/'
INPUT_SIZE = (473,473)

# colour map
COLORS = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
def get_lip_palette():
    palette = [0,0,0,
            128,0,0,
            255,0,0,
            0,85,0,
            170,0,51,
            255,85,0,
            0,0,85,
            0,119,221,
            85,85,0,
            0,85,85,
            85,51,0,
            52,86,128,
            0,128,0,
            0,0,255,
            51,170,221,
            0,255,255,
            85,255,170,
            170,255,85,
            255,255,0,
            255,170,0]
    return palette               
def get_palette(num_cls):
  """ Returns the color map for visualizing the segmentation mask.

  Inputs:
    =num_cls=
      Number of classes.

  Returns:
      The color map.
  """
  n = num_cls
  palette = [0] * (n * 3)
  for j in range(0, n):
    lab = j
    palette[j * 3 + 0] = 0
    palette[j * 3 + 1] = 0
    palette[j * 3 + 2] = 0
    i = 0
    while lab:
      palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
      palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
      palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
      i += 1
      lab >>= 3
  return palette

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="CE2P Network")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--dataset", type=str, default='val',
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")

    return parser.parse_args()

def valid(model, valloader, input_size, num_samples, gpus):
    model.eval()

    parsing_preds = np.zeros((num_samples, input_size[0], input_size[1]),
                             dtype=np.uint8)

    scales = np.zeros((num_samples, 2), dtype=np.float32)
    centers = np.zeros((num_samples, 2), dtype=np.int32)

    idx = 0
    interp = torch.nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    with torch.no_grad():
        for index, batch in enumerate(valloader):
            image, meta = batch
            num_images = image.size(0)
            if index % 10 == 0:
                print('%d  processd' % (index * num_images))

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            scales[idx:idx + num_images, :] = s[:, :]
            centers[idx:idx + num_images, :] = c[:, :]
            #====================================================================================
            org_img = image.numpy()
            normal_img = org_img
            flipped_img = org_img[:,:,:,::-1]
            fused_img = np.concatenate( (normal_img,flipped_img), axis=0 )
            outputs = model( torch.from_numpy(fused_img).cuda())
            prediction = interp( outputs[0][-1].cpu()).data.numpy().transpose(0, 2, 3, 1) #N,H,W,C
            single_out = prediction[:num_images,:,:,:]
            single_out_flip = np.zeros( single_out.shape )
            single_out_tmp = prediction[num_images:, :,:,:]
            for c in range(14):
                single_out_flip[:,:, :, c] = single_out_tmp[:, :, :, c]
            single_out_flip[:, :, :, 14] = single_out_tmp[:, :, :, 15]
            single_out_flip[:, :, :, 15] = single_out_tmp[:, :, :, 14]
            single_out_flip[:, :, :, 16] = single_out_tmp[:, :, :, 17]
            single_out_flip[:, :, :, 17] = single_out_tmp[:, :, :, 16]
            single_out_flip[:, :, :, 18] = single_out_tmp[:, :, :, 19]
            single_out_flip[:, :, :, 19] = single_out_tmp[:, :, :, 18]
            single_out_flip           = single_out_flip[:, :, ::-1, :]  
            # Fuse two outputs
            single_out = ( single_out+single_out_flip ) / 2
            parsing_preds[idx:idx + num_images, :, :] = np.asarray(np.argmax(single_out, axis=3), dtype=np.uint8)
            #====================================================================================
            # outputs = model(image.cuda())
            # if gpus > 1:
            #     for output in outputs:
            #         parsing = output[0][-1]
            #         nums = len(parsing)
            #         parsing = interp(parsing).data.cpu().numpy()
            #         parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
            #         parsing_preds[idx:idx + nums, :, :] = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)

            #         idx += nums
            # else:
            #     parsing = outputs[0][-1]
            #     parsing = interp(parsing).data.cpu().numpy()
            #     parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
            #     parsing_preds[idx:idx + num_images, :, :] = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)

            idx += num_images

    parsing_preds = parsing_preds[:num_samples, :, :]


    return parsing_preds, scales, centers

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    gpus = [int(i) for i in args.gpu.split(',')]

    h, w = map(int, args.input_size.split(','))
    
    input_size = (h, w)

    model = Res_Deeplab(num_classes=args.num_classes)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    lip_dataset = LIPDataSet(args.data_dir, 'val', crop_size=input_size, transform=transform)
    num_samples = len(lip_dataset)

    valloader = data.DataLoader(lip_dataset, batch_size=args.batch_size * len(gpus),
                                shuffle=False, pin_memory=True)

    restore_from = args.restore_from

    state_dict = model.state_dict().copy()
    state_dict_old = torch.load(restore_from)

    for key, nkey in zip(state_dict_old.keys(), state_dict.keys()):
        if key != nkey:
            # remove the 'module.' in the 'key'
            state_dict[key[7:]] = deepcopy(state_dict_old[key])
        else:
            state_dict[key] = deepcopy(state_dict_old[key])

    model.load_state_dict(state_dict)

    model.eval()
    model.cuda()

    parsing_preds, scales, centers = valid(model, valloader, input_size, num_samples, len(gpus))

    #=================================================================
    # list_path = os.path.join(args.data_dir, args.dataset + '_id.txt')
    # val_id = [i_id.strip() for i_id in open(list_path)]
    # pred_root = os.path.join( args.data_dir, 'pred_parsing')
    # if not os.path.exists( pred_root ):
    #     os.makedirs( pred_root )
    # palette = get_lip_palette()
    # output_parsing = parsing_preds
    # for i in range( num_samples ):
    #     output_image = PILImage.fromarray( output_parsing[i] )
    #     output_image.putpalette( palette )
    #     output_image.save( os.path.join( pred_root, str(val_id[i])+'.png'))    
    #=================================================================

    mIoU = compute_mean_ioU(parsing_preds, scales, centers, args.num_classes, args.data_dir, input_size)

    print(mIoU)

if __name__ == '__main__':
    main()
