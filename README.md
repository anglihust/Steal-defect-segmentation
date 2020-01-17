# Steal-defect-segmentation
Multiple segmentation networks for kaggle Steal defect segmentation:
https://www.kaggle.com/c/severstal-steel-defect-detection

# Segmentation Models
- SE uet
- Effcientnet0-unet
- Effcientnet1-unet
- HRnet
- Resnet(classfication)

# Dependencies 
- effcientnet
- keras
- bunch
- albumentations
- cv2
- glob

# How to use it
- Put images for training and testing in folders data\train_images and data\test_images
- Change "model_name" in config.json to train different segmentation models
- Change "cl_clean" in config.json to determine whether use classfication network to remove false positive segmentation
