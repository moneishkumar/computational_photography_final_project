# computational_photography_final_project

This is the repo for final project:
To run the code
```
python main.py --image <path to image> --kernels <path to calibrated kernels> --segmentation_mask <path to segmentation masks>
```

Kernels are list of matrices and segmentation mask are list of bounding boxes for RoIs they are stored as '.npy'


# Important files

- presentation_video.mp4: Presentation video
- report.pdf : report for the project
- Computational_Photography_Project_presentation.pdf: presentation slides
- data : This has the images used in the report and also the calibrated kernels used.

## Prerequisites

This repo requires two submodules:

MMdetection and MMsegmentation 

- https://github.com/open-mmlab/mmdetection
- https://github.com/open-mmlab/mmsegmentation

Use the object detection in mmdetection to first detect the bounding boxes .
Then use the cropped and pass it to the image segmentation model to get the segmentation mask.
