# Comp-541-final-project



--> Pre-trained weights are used to test the model. (RPNLast.mat)

--> One input image file is added to save the storage, rest can be found at: http://dss.cs.princeton.edu/Release/sunrgbd_dss_data/SUNRGBD/kv2/kinect2data/

--> TSDF of each image is calculated running tsdf.h, tsdf_kernels.cu, and tsdf.cu.

--> RPN.jl includes the model, produces bounding boxes for each image, and calculates the accuracy with ground truth values.

--> NMS (Non-maximum supression) is used for removing the unnecessary boxes
