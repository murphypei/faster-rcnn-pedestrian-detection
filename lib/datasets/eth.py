#! -*- coding:utf-8 -*-

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


class eth(imdb):
    def __init__(self,
                 version,
                 image_set,
                 devkit_path="caltech-pedestrian-dataset-converter"):
        imdb.__init__(self, 'caltech_pedestrian_' + image_set)
        self.version = version

        self.config = {
            "include_all_classes": False,
            "include_background": False
        }
        self._image_set = image_set
        self._devkit_path = self._get_default_path(
        ) if devkit_path is None else devkit_path
        self._data_path = os.path.join("data", self._devkit_path, 'INRIA')
        self._classes = (
            '__background__',  # always index 0
            'person')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        annotation_path = os.path.join(self._data_path,
                                       "INRIA_annotations.json")
        assert os.path.exists(
            annotation_path), 'Annotation path does not exist.: {}'.format(
                annotation_path)

        self._annotation = json.load(open(annotation_path))

        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        self._salt = str(uuid.uuid4())
        '''
        # Caltech Pedestrain specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}
        '''
        # not usre if I should keep this line
        # assert os.path.exists(self._devkit_path), \
        #        'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(
            self._data_path), 'Path does not exist: {}'.format(
                self._data_path)

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
                                  index + self._image_ext)
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(
            image_path)
        return image_path


# Strategy: get the index from annotation dictionary

    def _load_image_set_list(self):
        image_set_file = os.path.join(self._data_path,
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        f = open(image_set_file)
        return [line.strip() for line in f]

    def all_index(self, image_set_list):
        image_index = []
        for set_num in self._annotation:
            if int(set_num[3:]) in image_set_list:
                print("Loading: {}".format(set_num))
                for v_num in self._annotation[set_num]:
                    for frame_num in self._annotation[set_num][v_num][
                            "frames"]:
                        image_index.append("{}_{}_{}".format(
                            set_num, v_num, frame_num))

        return image_index

    def person_class_index(self, image_set_list):
        image_index = self.all_index(image_set_list)
        target_index = []
        for image_name in image_index:
            set_num, v_num, frame_num = image_name.split("_")
            boxes = self._annotation[set_num][v_num]["frames"][frame_num]
            if any(box["lbl"] == "person" for box in boxes):
                target_index.append(image_name)

        return target_index

    def reasonable_index(self, image_set_list):
        def verify_person_class(box):
            return box['lbl'] == 'person'

        def verify_reasonable(box):
            def verity_bnds(pos):
                bnds = [5, 5, 635, 475]
                return pos[0] >= bnds[0] and pos[0] + pos[2] <= bnds[2] and pos[1] >= bnds[1] and pos[1] + pos[3] <= bnds[3]

            height_min = 50
            visiable_min = .65
            pos = box['pos']
            pos_v = box['posv']
            occl = box['occl']
            # label = box["lbl"]

            pos_area = pos[2] * pos[3]

            if occl == 0 or not hasattr(
                    pos_v, '__iter__') or all(x == 0 for x in pos_v):

                visiable_ratio = 1
            else:
                pos_v_area = pos_v[2] * pos_v[3]
                visiable_ratio = (pos_v_area / pos_area)

            return verify_person_class(
                box
            ) and visiable_ratio > visiable_min and pos[3] > height_min and verity_bnds(
                pos)

        image_index = self.person_class_index(image_set_list)
        target_index = []

        for image_name in image_index:
            set_num, v_num, frame_num = image_name.split("_")
            boxes = self._annotation[set_num][v_num]["frames"][frame_num]
            if any(verify_reasonable(box) for box in boxes):
                target_index.append(image_name)

        return target_index

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt

        image_set_list = [
            int(image_set_num)
            for image_set_num in self._load_image_set_list()
        ]

        image_path = os.path.join(self._data_path, 'images')
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        image_index = []

        print(image_set_list)

        method_mapper = {
            "reasonable": self.reasonable_index,
            "all": self.all_index,
            "person": self.person_class_index
        }

        image_index = method_mapper[self.version](image_set_list)

        return image_index

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

        gt_roidb = [
            self._load_caltech_annotation(
                index)  #This line is crucially  important 
            for index in self.image_index
        ]
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
        filename = os.path.abspath(
            os.path.join(cfg.DATA_DIR, 'selective_search_data',
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

    # Assign negtaive example to __background__ as whole image
    def _load_caltech_annotation(self, index):
        def verify_person_class(box):
            return box['lbl'] == 'person'

        def verify_reasonable(box):
            def verity_bnds(pos):
                bnds = [5, 5, 635, 475]
                return pos[0] >= bnds[0] and pos[0] + pos[2] <= bnds[2] and pos[1] >= bnds[1] and pos[1] + pos[3] <= bnds[3]

            height_min = 50
            visiable_min = .65
            pos = box['pos']
            pos_v = box['posv']
            occl = box['occl']
            # label = box["lbl"]

            pos_area = pos[2] * pos[3]

            if occl == 0 or not hasattr(
                    pos_v, '__iter__') or all(x == 0 for x in pos_v):

                visiable_ratio = 1
            else:
                pos_v_area = pos_v[2] * pos_v[3]
                visiable_ratio = (pos_v_area / pos_area)

            return verify_person_class(
                box
            ) and visiable_ratio > visiable_min and pos[3] > height_min and verity_bnds(
                pos)

        verify_all = lambda box: True
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC format.
        """
        filename = os.path.join(self._data_path, "annotation.json")
        # annotation = json.load(open(filename))
        set_num, v_num, frame_num = index.split("_")
        bboxes = self._annotation[set_num][v_num]["frames"][frame_num]
        print(len(bboxes))

        verify_methods = {
            "person": verify_person_class,
            "reasonable": verify_reasonable,
            "all": verify_all
        }
        verify_method = verify_methods[self.version]

        bboxes = [bbox for bbox in bboxes if verify_method(bbox)]
        if not verify_reasonable(bbox):
            print("Filter out non {} boxes".format(self.version))

        num_objs = len(bboxes)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        # Becareful about the coordinate format
        # Load object bounding boxes into a data frame.

        cls = 1

        # This is possitive example
        for ix, bbox in enumerate(bboxes):

            x1 = float(bbox['pos'][0])
            y1 = float(bbox['pos'][1])
            x2 = float(bbox['pos'][0] + bbox['pos'][2])
            y2 = float(bbox['pos'][1] + bbox['pos'][3])
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = 1  #Must be pedestrian
            overlaps[ix, cls] = 1.0
            if (y2 - y1) < 50:
                print("Oops!")
            seg_areas[ix] = (x2 - x1) * (y2 - y1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {
            'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas
        }

    def _get_caltech_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._salt + self._image_set + '_{:s}.txt'
        path = os.path.join(self._devkit_path, 'results', filename)
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
            return [atoi(c) for c in re.split('(\d+)', text)]

        def insert_frame(target_frames,
                         file_path,
                         start_frame=30,
                         frame_rate=30):
            file_name = file_path.split("/")[-1]
            set_num, v_num, frame_num = file_name[:-4].split("_")
            if int(frame_num) >= start_frame and int(
                    frame_num) % frame_rate == 0:
                target_frames.setdefault(set_num, {}).setdefault(
                    v_num, []).append(file_path)
                return 1
            else:
                return 0

        def detect(file_path, NMS_THRESH=0.3):
            im = cv2.imread(file_path)
            scores, boxes = im_detect(net, im)
            cls_scores = scores[:, 1]
            cls_boxes = boxes[:, 4:8]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(
                np.float32)
            keep = nms(dets, NMS_THRESH)
            return dets[keep, :]

        def get_target_frames(image_set_list, image_path):
            target_frames = {}
            total_frames = 0
            for set_num in image_set_list:
                file_pattern = "{}/set{}_V*".format(image_path, set_num)
                file_list = sorted(glob.glob(file_pattern), key=natural_keys)
                for file_path in file_list:
                    total_frames += insert_frame(target_frames, file_path)

            return target_frames, total_frames

        def detection_to_file(target_path,
                              v_num,
                              file_list,
                              detect,
                              total_frames,
                              current_frames,
                              max_proposal=100,
                              thresh=0):
            timer = Timer()
            w = open("{}/{}.txt".format(target_path, v_num), "w")
            for file_index, file_path in enumerate(file_list):
                file_name = file_path.split("/")[-1]
                set_num, v_num, frame_num = file_name[:-4].split("_")

                timer.tic()
                dets = detect(file_path)

                timer.toc()

                print('Detection Time:{:.3f}s  {}/{} images'.format(
                    timer.average_time, current_frames + file_index + 1,
                    total_frames))

                inds = np.where(dets[:, -1] >= thresh)[0]
                for i in inds:
                    bbox = dets[i, :4]
                    score = dets[i, -1]

                    x = bbox[0]
                    y = bbox[1]
                    width = bbox[2] - x
                    length = bbox[3] - y
                    w.write("{},{},{},{},{},{}\n".format(
                        frame_num, x, y, width, length, score * 100))

            w.close()
            print("Evalutaion file {} has been writen".format(w.name))
            return file_index + 1

        model_name = net.name
        output_path = os.path.join(self._data_path, "res", self.version,
                                   model_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        image_set_list = self._load_image_set_list()

        image_path = os.path.join(self._data_path, 'images')
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(
            image_path)
        target_frames, total_frames = get_target_frames(
            image_set_list, image_path)

        current_frames = 0
        for set_num in target_frames:
            target_path = os.path.join(output_path, set_num)
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            for v_num, file_list in target_frames[set_num].items():
                current_frames += detection_to_file(
                    target_path, v_num, file_list, detect, total_frames,
                    current_frames)

    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(self._devkit_path, 'VOC' + self._year,
                                'Annotations', '{:s}.xml')
        imagesetfile = os.path.join(self._devkit_path, 'VOC' + self._year,
                                    'ImageSets', 'Main',
                                    self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename,
                annopath,
                imagesetfile,
                cls,
                cachedir,
                ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print '-----------------------------------------------------'
        print 'Computing results with the official MATLAB eval code.'
        print '-----------------------------------------------------'
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
               .format(self._devkit_path,
                       self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_caltech_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':

    from datasets.eth import eth
    d = eth("trainval")
    res = d.roidb
    from IPython import embed
    embed()
