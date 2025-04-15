import torch
import torch.nn.functional as F

def custom_collate_fn(batch):
    # Unzip the batch. Each element of batch is assumed to be a tuple.
    images, labels, *others = zip(*batch)
    target_size = (256, 256)  # Desired output size
    
    processed_imgs = []
    for img in images:
        # If the image is already batched (i.e. 4D tensor), assume it's [B, C, H, W]
        if img.dim() == 4:
            # Instead of treating the whole tensor as one sample,
            # break it up into its individual images.
            for i in range(img.size(0)):
                single_img = img[i]
                # If the single image is 2D, add a channel dimension
                if single_img.dim() == 2:
                    single_img = single_img.unsqueeze(0)
                processed_imgs.append(F.adaptive_avg_pool2d(single_img, target_size))
        else:
            # For unbatched images
            if img.dim() == 2:
                img = img.unsqueeze(0)
            processed_imgs.append(F.adaptive_avg_pool2d(img, target_size))
    
    # Now stack along the batch dimension:
    images_tensor = torch.stack(processed_imgs)
    
    processed_labels = []
    for lbl in labels:
        if lbl.dim() == 4:
            for i in range(lbl.size(0)):
                single_lbl = lbl[i]
                if single_lbl.dim() == 2:
                    single_lbl = single_lbl.unsqueeze(0)
                processed_labels.append(F.adaptive_avg_pool2d(single_lbl, target_size))
        else:
            if lbl.dim() == 2:
                lbl = lbl.unsqueeze(0)
            processed_labels.append(F.adaptive_avg_pool2d(lbl, target_size))
    
    labels_tensor = torch.stack(processed_labels)
    
    # Process additional fields (e.g., index, spatialWeights, maxDist)
    rest_processed = []
    for group in others:
        # If group elements are tensors, stack them; otherwise, convert to list.
        if isinstance(group[0], torch.Tensor):
            rest_processed.append(torch.stack(group))
        else:
            rest_processed.append(list(group))
    
    return (images_tensor, labels_tensor) + tuple(rest_processed)
