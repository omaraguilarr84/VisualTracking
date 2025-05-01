All models with no label are trained on:
* 10 epochs
* * We are currently limited by time and computational resources
* 16 Batch Size
---
If the model weight name has "BS#", this means the Batch Size was altered to the number specified.

If the model weight name has "ER#", this means the Expansion Ratio in the Inverted Residual Blocks for the MobileNet models was altered to the number specified.

If the model weight has "AP", then the model implements Adaptive Average Pooling to account for variable image sizes across batches.

If the model weight has "ROI", then the model is trained on the autoROI-processed images for smaller but variable image sizes.