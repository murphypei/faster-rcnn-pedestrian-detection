
#This is negative_ignore version of imdb class for Caltech Pedestrian dataset


import re
import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval
from fast_rcnn.config import cfg
import json
from os import listdir
from os.path import isfile, join
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import glob
import cv2


# Public functions: filter for all dataset.

# For boxes with certain Label, default is person_class only
def label_filter(box, label="person"):
    return box['lbl'] == label

# For boxes with a specified boundry, the default values arefrom 
def boundry_filter(box, bnds = {'xmin':5, 'ymin':5, 'xmax':635, 'ymax':475}):
    x1 = box['pos'][0]
    y1 = box['pos'][1]
    width = box['pos'][2]
    height = box['pos'][3]
    x2 = x1 + width
    y2 = y1 + height

    validity =  x1 >= bnds['xmin'] and \
                x2 <= bnds['xmax'] and \
                y1 >= bnds['ymin'] and \
                y2 <= bnds['ymax'] 

    return validity

# For boxes higher than a speifcied height
def height_filter(box, height_range = {'min':50, 'max': float('inf')}):
    height = box['pos'][3]
    validity = height >= height_range['min'] and \
               height < height_range['max']
    return validity

# For boxes more visible than a speifcied range
def visibility_filter(box, visible_range = {'min': 0.65, 'max': float('inf')}):
    occluded = box['occl']

    # A dirty condition to deal with the ill-formatted data.
    if occluded == 0 or \
       not hasattr(box['posv'], '__iter__') or \
       all([v==0 for v in box['posv']]):

        visiable_ratio = 1

    else:
        width = box['pos'][2]
        height = box['pos'][3]
        area = width * height   

        visible_width = box['posv'][2]
        visible_height = box['posv'][3]
        visible_area = visible_width * visible_height

        visiable_ratio = visible_area / area


    validity = visiable_ratio  >= visible_range['min'] and \
           visiable_ratio  <= visible_range['max']

    return validity


    height = box['pos'][3]
    validity = height >= height_range['min'] and \
               height < height_range['max']
    return validity

# For reasonable subset
def reasonable_filter(box):
    label = "person"
    validity = box['lbl'] == 'person' and\
               boundry_filter(box) and\
               height_filter(box) and \
               visibility_filter(box)

    return validity

# For all dataset
true_filter = lambda box: True


# caltech imdb
class caltech(imdb):
    def __init__(self, version, image_set, devkit_path="caltech-pedestrian-dataset-converter"):
        imdb.__init__(self,'caltech_pedestrian_' + image_set)
        self.version = version
        
        self.config = {"include_all_classes":False, "include_background": False}
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join("data",self._devkit_path, 'data')
        self._classes = ('__background__', # always index 0
                         'person')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        annotation_path = os.path.join(self._data_path, "annotations.json")
        assert os.path.exists(annotation_path), \
                'Annotation path does not exist.: {}'.format(annotation_path)
         

        self._annotation = json.load(open(annotation_path))
        
        
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        self._salt = str(uuid.uuid4())
    
        #not usre if I should keep this line
        #assert os.path.exists(self._devkit_path), \
        #        'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
       
        image_path = os.path.join(self._data_path, 'images',
                                  index+self._image_ext )
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path
#Strategy: get the index from annotation dictionary 
    
   
    def _load_image_set_list(self):
        image_set_file = os.path.join(self._data_path,
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        f = open(image_set_file)
        return  [line.strip() for line in f]
    
    
       
    def all_index(self, image_set_list):
        image_index = []        
        for set_num in self._annotation:
            if int(set_num[3:]) in image_set_list:
                print("Loading: {}".format(set_num))
                for v_num in self._annotation[set_num]:
                    for frame_num in self._annotation[set_num][v_num]["frames"]:
                        image_index.append("{}_{}_{}".format(set_num, v_num, frame_num))
                     
        return image_index                   
                    

    def reasonable_index(self, image_set_list):
        

        image_index = self.person_class_index(image_set_list)
        target_index = []

        for image_name in image_index :
            set_num, v_num, frame_num =  image_name.split("_")
            boxes = self._annotation[set_num][v_num]["frames"][frame_num]
            if any(reasonable_filter(box) for box in boxes):
                target_index.append(image_name)
                                
        return target_index                   
     

    def _load_image_set_index(self):
      
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
       
        image_set_list = [int(image_set_num) for image_set_num  in self._load_image_set_list()]
        
        
        image_path = os.path.join(self._data_path, 'images')
        assert os.path.exists( image_path), \
                'Path does not exist: {}'.format( image_path)
        image_index = []

        print(image_set_list)
        
                                
        filter_mapper = {"reasonable": reasonable_filter, "all": true_filter, "person_class":\
                         label_filter}
        
        box_filter = filter_mapper[self.version]
                        
        
        all_index = self.all_index(image_set_list)
        target_index = []

        for image_name in all_index  :
            set_num, v_num, frame_num =  image_name.split("_")
            boxes = self._annotation[set_num][v_num]["frames"][frame_num]
            if any(box_filter(box) for box in boxes):
                target_index.append(image_name)
        
       
       
        return target_index
    

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.caltech_imdb._load_caltech_annotation

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_caltech_annotation(index)    #This line is crucially  important 
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)
    
    
    #Assign negtaive example to __background__ as whole image
    def _load_caltech_annotation(self, index):

        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, "annotation.json")
        #annotation = json.load(open(filename))
        set_num, v_num, frame_num = index.split("_")
        bboxes = self._annotation[set_num][v_num]["frames"][frame_num]
        
        
        verify_methods = {"person_class_only":label_filter, "reasonable":reasonable_filter, "all": lambda box: True  }
        verify_method = verify_methods[self.version]
        original_len = len(bboxes)
        bboxes = [bbox for bbox in bboxes if verify_method(bbox) ]
        num_objs = len(bboxes)
        if original_len > num_objs:
            print("Filter out {} non-{} boxes".format(original_len - num_objs, self.version))
        #if not verify_reasonable(bbox):
            #print("Filter out non {} boxes".format(self.version))
          

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        #Becareful about the coordinate format
        # Load object bounding boxes into a data frame.
        

        cls = 1
        # This is possitive example
        for ix, bbox in enumerate(bboxes):
            

            x1 = float(bbox['pos'][0])
            y1 = float(bbox['pos'][1])
            x2 = float(bbox['pos'][0] + bbox['pos'][2])
            y2 = float(bbox['pos'][1] + bbox['pos'][3])
            assert(self.version != "reasonable" or (y2 - y1) >= 50, \
                   "Bounding box is too samll, Reasonable Filter is not working.")
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = 1  #Must be pedestrian
            overlaps[ix, cls] = 1.0
            
             
            seg_areas[ix] = (x2 - x1) * (y2 - y1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}


    def _get_caltech_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._salt + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            filename)
        return path
    # This method write results files into Evaluation toolkit format
    def _write_caltech_results_file(self, net):
         
        #Insert my code in the following space
        
        # The follwing nested fucntions are for smart sorting
        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            '''
            alist.sort(key=natural_keys) sorts in human order
            http://nedbatchelder.com/blog/200712/human_sorting.html
            (See Toothy's implementation in the comments)
            '''
            return [ atoi(c) for c in re.split('(\d+)', text) ]
        
        def insert_frame(target_frames, file_path,start_frame=29, frame_rate=30):
            file_name = file_path.split("/")[-1]
            set_num, v_num, frame_num = file_name[:-4].split("_")
            if int(frame_num) >= start_frame and int(frame_num) % frame_rate == 29:
                target_frames.setdefault(set_num,{}).setdefault(v_num,[]).append(file_path)
                return 1
            else:
                return 0
                 
          
        def detect(file_path,  NMS_THRESH = 0.3):
            im = cv2.imread(file_path)
            scores, boxes = im_detect(net, im)
            cls_scores = scores[: ,1]
            cls_boxes = boxes[:, 4:8]
            dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            return dets[keep, :]
             
        
        def get_target_frames(image_set_list,  image_path):
            target_frames = {}
            total_frames = 0 
            for set_num in image_set_list:
                file_pattern = "{}/set{}_V*".format(image_path,set_num)
                file_list = sorted(glob.glob(file_pattern), key=natural_keys)
                for file_path in file_list:
                    total_frames += insert_frame(target_frames, file_path)
                
            return target_frames, total_frames 
        
        def detection_to_file(target_path, v_num, file_list, detect,total_frames, current_frames, max_proposal=100, thresh=0):
            timer = Timer()
            w = open("{}/{}.txt".format(target_path, v_num), "w")
            for file_index, file_path in enumerate(file_list):
                file_name = file_path.split("/")[-1]
                set_num, v_num, frame_num = file_name[:-4].split("_")
                frame_num = str(int(frame_num) +1)
                
                timer.tic()
                dets = detect(file_path)
               
                timer.toc()
                 
                print('Detection Time:{:.3f}s on {}  {}/{} images'.format(timer.average_time,\
                                                       file_name ,current_frames+file_index+1 , total_frames))
                
                             
                inds = np.where(dets[:, -1] >= thresh)[0]     
                for i in inds:
                    bbox = dets[i, :4]
                    score = dets[i, -1]
            
                    x = bbox[0]
                    y = bbox[1] 
                    width = bbox[2] - x 
                    length =  bbox[3] - y
                    w.write("{},{},{},{},{},{}\n".format(frame_num, x, y, width, length, score*100))
                    
               
            w.close()
            print("Evalutaion file {} has been writen".format(w.name))   
            return file_index + 1
               

                        
        model_name = net.name
        output_path = os.path.join(self._data_path,"res" , self.version, model_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)       
            
        
        image_set_list = self._load_image_set_list()

        
        image_path = os.path.join(self._data_path, 'images')
        assert os.path.exists(image_path),'Path does not exist: {}'.format(image_path)
        target_frames, total_frames = get_target_frames(image_set_list,  image_path)

        
        current_frames = 0
        for set_num in target_frames:
            target_path = os.path.join(output_path, set_num)
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            for v_num, file_list in target_frames[set_num].items():
                current_frames += detection_to_file(target_path, v_num, file_list, detect, total_frames, current_frames)
                


if __name__ == '__main__':
    from datasets.pascal_voc import caltech
    d = caltech("trainval")
    res = d.roidb
    from IPython import embed; embed()
