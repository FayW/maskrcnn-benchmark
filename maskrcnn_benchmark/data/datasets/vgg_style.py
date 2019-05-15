import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
import torch
import torch.utils.data
from PIL import Image
import sys
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from maskrcnn_benchmark.structures.bounding_box import BoxList

############################################################
#  Bounding Boxes
############################################################

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([x1, y1, x2, y2])
    return boxes.astype(np.int32)



class MultihandDataset(torch.utils.data.Dataset):
    CLASSES = (
        "__background__ ",
        "people",
    )
    def __init__(self, data_dir,transforms=None):
        """Load a subset of the Multihand dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        #super(MultihandDataset, self).__init__(data_dir)
        # Train or validation dataset?
        self.data_dir = os.path.join(data_dir)
        self.transforms = transforms
        self.image_info = []
        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(self.data_dir, "via_region_data.json")))
        #annotations = list(annotations)[0]
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(self.data_dir, a['filename'])
            # sometimes
            if not os.path.isfile(image_path):
                print('skip non-exists image ',image_path)
                continue
            try:
                image = skimage.io.imread(image_path)
            except:
                print('skip image {}'.format(image_path))
                continue
            height, width = image.shape[:2]
            image_instance = {
                #"id": image_id,
                #"source": source,
                "path": image_path,
                "width":width,
                "height":height,
                "polygons":polygons,
                
            }
            mask_OK=True
            for i, p in enumerate(polygons):
                # Get indexes of pixels inside the polygon and set them to 1
                #r, c = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
                if max(p['all_points_y'])>=height or max(p['all_points_x'])>=width:
                    print('bad mask {}'.format(i))
                    mask_OK=False
                    break
            if mask_OK==True:
                self.image_info.append(image_instance)
        
        cls = MultihandDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
                
    def __len__(self):
        return len(self.image_info)

    def load_image(self, image_idx):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_idx]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def __getitem__(self, index):
        #img = self.load_image(index)
        img = Image.open(self.image_info[index]['path']).convert("RGB")
        target = self.load_image_gt(index)
        target = target.clip_to_image(remove_empty=True)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index


    def load_mask(self, image_idx):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        #image_info = self.image_info[image_id]
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_idx]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            try:
                mask[rr, cc, i] = 1
            except:
                print('h {} w {} row {} col {}'.format(info["height"],info["width"],rr,cc))

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def load_polygons(self, image_idx):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        #image_info = self.image_info[image_id]
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        polygons = self.image_info[image_idx]["polygons"]
        polys_pts=[]
        for poly in polygons:
            xpts = poly["all_points_x"]
            ypts = poly["all_points_y"]
            poly_pts=[]
            for i in range(len(xpts)):
                poly_pts.extend([xpts[i],ypts[i]])
            polys_pts.append([poly_pts])
        return polys_pts
   


    def load_image_gt(self, idx):
        """Load and return ground truth data for an image (image, mask, bounding boxes).

        augment: (deprecated. Use augmentation instead). If true, apply random
            image augmentation. Currently, only horizontal flipping is offered.
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
            For example, passing imgaug.augmenters.Fliplr(0.5) flips images
            right/left 50% of the time.
        use_mini_mask: If False, returns full-size masks that are the same height
            and width as the original image. These can be big, for example
            1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
            224x224 and are generated by extracting the bounding box of the
            object and resizing it to MINI_MASK_SHAPE.

        Returns:
        image: [height, width, 3]
        shape: the original shape of the image before resizing and cropping.
        class_ids: [instance_count] Integer class IDs
        bbox: [instance_count, (y1, x1, y2, x2)]
        mask: [height, width, instance_count]. The height and width are those
            of the image unless use_mini_mask is True, in which case they are
            defined in MINI_MASK_SHAPE.
        """
        # Load image and mask
        mask, class_ids = self.load_mask(idx)
        polygon = self.load_polygons(idx)
        img = Image.open(self.image_info[idx]['path']).convert("RGB")

        # Note that some boxes might be all zeros if the corresponding mask got cropped out.
        # and here is to filter them out
        _idx = np.sum(mask, axis=(0, 1)) > 0
        mask = mask[:, :, _idx]
        class_ids = class_ids[_idx]
        # Bounding boxes. Note that some boxes might be all zeros
        # if the corresponding mask got cropped out.
        # bbox: [num_instances, (y1, x1, y2, x2)]
        bbox = extract_bboxes(mask)
        target = BoxList(bbox, img.size, mode="xyxy")
        class_ids = torch.tensor(class_ids)
        target.add_field("labels", class_ids)
        #print('POLY {}'.format(polygon))
        masks = SegmentationMask(polygon, img.size, mode='poly')
        target.add_field("masks", masks)
        # Active classes
        # Different datasets have different classes, so track the
        # classes supported in the dataset of this image.
        #active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
        #source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
        #active_class_ids[source_class_ids] = 1

        # Resize masks to smaller size to reduce memory usage
        #if use_mini_mask:
        #    mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

        # Image meta data
        #image_meta = compose_image_meta(image_id, original_shape, image.shape, window, scale, active_class_ids)

        return target
        
    def get_img_info(self, index):
        return {"height": self.image_info[index]['height'], "width": self.image_info[index]['width']}
        
