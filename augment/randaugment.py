import random
import numpy as np

import augment.randaugment_ops.policies as found_policies
import augment.randaugment_ops.augmentation_transforms as transform

MEAN, STD = transform.get_mean_and_std()
POLICIES = found_policies.randaug_policies()

def randaugment(image):
    image = image / 255.
    image = (image - MEAN) / STD
    
    chosen_policy = random.choice(POLICIES)
    aug_image = transform.apply_policy(chosen_policy, image)
    aug_image = transform.cutout_numpy(aug_image)
    
    aug_image = (aug_image * STD) + MEAN
    aug_image *= 255.
    
    return aug_image.astype(np.float32)
