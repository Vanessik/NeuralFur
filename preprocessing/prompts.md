# ChatGPT Prompts for Fur Annotations

To obtain annotations for a new animal, send two images (frontal and side views) to ChatGPT and use the following prompts.

## 1. Length annotations

> Here, you could see images of an animal. Could you estimate the accurate fur length in cm for each part of the animal: "leg_front", "leg_rear", "paws", "front_paws", "belly", "neck", "face", "ears", "under_tail", "tail", "body", "paw_pads", "inner_earcanal", "eyes", "nosetip"? Also, does this animal have some fur near the neck that grows significantly beyond the underlying body? If so, could you add the estimates for length and fur thickness as well for "mane" and include it at the end of the previous part names; otherwise, do not include it. Is the ear canal visible in the image? If the inner ear canal is not visible, use the same value for length as for the outer ear. Please provide results in dict format, where the key is part name and the value is your estimate.

## 2. Effective fur thickness

> I want to create a furless animal. To do that, I have a 3D model of an animal with fur, from which I want to subtract the region covered by fur for each part. Could you provide the number of effective fur thickness for each part that I need to subtract from the full geometry? Please provide results in dict format where the key is part name and the value is your estimate.

Then verify the estimates:

> Are you sure of the initial length estimations and fur thickness to abstract? Could you double-check with the image? Please provide results in a several-dict format where the key is part name and the value is your estimate.

## 3. Hair growing direction

> Could you also estimate what the fur growing direction looks like in 3d space? I want to know which parts are oriented along the gravity vector and which are against. Also, if fur for regions grows from left to right or from right to left. I have the following coordinate system: x from the right side of animal towards the left, y - opposite to gravity direction, and z from back to the front of the animal. The second image is aligned with the 3D coordinate system. I want for each part: "leg_front", "leg_rear", "paws", "front_paws", "belly", "neck", "face", "ears", "under_tail", "tail", "body", "paw_pads", "inner_earcanal", "eyes", "nosetip", "mane" (if appeared) obtains a vector that defines the approximate growing direction in the coordinate system. Please double-check the proposed directions several times.

## 4. Eyeballs distance

> Could you measure the distance between eyeballs in cm? You could also use the real-world understanding of this animal class.

## Usage

Paste the obtained length and thickness dicts into `mapping_length` and `effective_fur_thickness_cm` in `src/animal_config.py`. Paste the gravity directions into `mapping_gravity` in the YAML config at `submodules/GaussianHaircut/src/arguments/`. Set `eye_dists_VQA` to the eyeball distance value.
