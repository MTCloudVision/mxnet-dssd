# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import numpy as np
from dataset.imdb import Imdb
from dataset.pycocotools.coco import COCO
if True:
    with open('/data2/yry/rcnn/data/coco/labels.txt','r') as f:
        lines = f.readlines()
    lines=[[x.split(',')[0],x.split(',')[1]] for x in lines]
    ori=[]
    tar=[]
    for i in lines:
        ori.append(int(i[0]))
        tar.append(int(i[1]))
    tar=[x-1 for x in tar]
    labeltrans_dict=dict(zip(ori,tar))

class Coco(Imdb):
    """
    Implementation of Imdb for MSCOCO dataset: https://http://mscoco.org

    Parameters:
    ----------
    anno_file : str
        annotation file for coco, a json file
    image_dir : str
        image directory for coco images
    shuffle : bool
        whether initially shuffle image list

    """
    def __init__(self, anno_file, image_dir, shuffle=True, names='mscoco.names'):
        print('load_coco!!!!!!!!!!!')
        assert os.path.isfile(anno_file), "Invalid annotation file: " + anno_file
        basename = os.path.splitext(os.path.basename(anno_file))[0]
        print('basename:',basename)
        super(Coco, self).__init__('coco_' + basename)
        self.image_dir = image_dir

        self.classes = self._load_class_names(names,
            os.path.join(os.path.dirname(__file__), 'names'))

        self.num_classes = len(self.classes)
        self._load_all(anno_file, shuffle)
        self.num_images = len(self.image_set_index)
#        self.num_images = len(self.image_set_index)
        print('coco self.num_images:',self.num_images)
        labeltrans='/data2/yry/rcnn/data/coco/labels.txt'
        with open(labeltrans,'r') as f:
            lines = f.readlines()
        lines=[[x.split(',')[0],x.split(',')[1]] for x in lines]
        ori=[]
        tar=[]
        for i in lines:
            ori.append(i[0])
            tar.append(i[1])
        self.labeltrans=dict(zip(ori,tar))


    def image_path_from_index(self, index):
        """
        given image index, find out full path

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        full path of this image
        """
        assert self.image_set_index is not None, "Dataset not initialized"
        name = self.image_set_index[index]
        image_file = os.path.join(self.image_dir, 'images', name)
        assert os.path.isfile(image_file), 'Path does not exist: {}'.format(image_file)
        print(image_file)
        return image_file

    def label_from_index(self, index):
        """
        given image index, return preprocessed ground-truth

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        ground-truths of this image
        """
        assert self.labels is not None, "Labels not processed"
        return self.labels[index]

    def _load_all(self, anno_file, shuffle):
        """
        initialize all entries given annotation json file

        Parameters:
        ----------
        anno_file: str
            annotation json file
        shuffle: bool
            whether to shuffle image list
        """
        image_set_index = []
        labels = []
        coco = COCO(anno_file)
        print(coco)
        img_ids = coco.getImgIds()
        #print (anno_file)
        subdir = anno_file.split('/')[-1].split('.')[0].split('_')[2]
        #print (subdir)
        print('img_ids_len:',len(img_ids))
        for img_id in img_ids:
            # filename
            image_info = coco.loadImgs(img_id)[0]
            filename = image_info["file_name"]
            #print(image_info)
            height = image_info["height"]
            width = image_info["width"]
            # label
            anno_ids = coco.getAnnIds(imgIds=img_id)
            #print(anno_ids)
            annos = coco.loadAnns(anno_ids)
            #print(annos)
            label = []
            for anno in annos:
                print(len(annos)) 
                cat_id = int(anno["category_id"])
                cat_id = labeltrans_dict[cat_id]
#                print (cat_id)                
                bbox = anno["bbox"]
                assert len(bbox) == 4
                xmin = float(bbox[0]) / width
                ymin = float(bbox[1]) / height
                xmax = xmin + float(bbox[2]) / width
                ymax = ymin + float(bbox[3]) / height
                label.append([cat_id, xmin, ymin, xmax, ymax, 0])
            if label:
                labels.append(np.array(label))
                image_set_index.append(os.path.join(subdir, filename))

        if shuffle:
            import random
            indices = range(len(image_set_index))
            random.shuffle(indices)
            image_set_index = [image_set_index[i] for i in indices]
            labels = [labels[i] for i in indices]
        # store the results
        self.image_set_index = image_set_index
        self.labels = labels
