import os
import cv2 
import argparse

import torch 
import torchvision
import torchvision.transforms as transforms

from utils import draw_segmentation_map, get_outputs

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='C:\\Users\\USER\\Desktop\\MrShahrullo\\Instance_segmentation\\big_buck_bunny_720p_5mb.mp4\\', help='path to the input video data')
parser.add_argument('-t', '--threshold', default=0.965, type=float, help='score threshold for discarding detection')
args = vars(parser.parse_args())

vid_path = args['input']

vidcap = cv2.VideoCapture(vid_path)

success, image = vidcap.read()

count = 0
newpath = 'C:\\Users\\USER\\Desktop\\MrShahrullo\\Instance_segmentation\\frames\\'
while success:
    if not os.path.exists(newpath):
        os.makedirs(newpath, exist_ok=True)
    # cv2.imwrite("/home/shahrullohon/Desktop/PROJECTS/SEGMENTATION/MaskR-CNN/frames/frame%d.jpg" % count, image) # save frame as JPEG file
    success, image = vidcap.read()    
    print('Read a new frame: ', success)

    # initialize the model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # transform the image data to tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # keep copy of original image
    orig_image = image.copy()

    # transform the data
    image = transform(image) 
    # add a batch dimension
    image = image.unsqueeze(0).to(device)

    # get the prediction
    masks, boxes, labels = get_outputs(image, model, args['threshold'])
    # get the final result
    result = draw_segmentation_map(orig_image, masks, boxes, labels)


    # visualize the image
    # cv2.imshow("Segmented image", result)
    # cv2.waitKey(1)

    cv2.imwrite("C:\\Users\\USER\\Desktop\\MrShahrullo\\Instance_segmentation\\frames\\frame%d.jpg" % count, result) 
    print("Result saved")

    count += 1

cv2.destroyAllWindows()



if __name__ == "__main__":

    print(f"Run successfully\nNumber of frames are: {count}")