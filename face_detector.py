#!/usr/bin/python3
# -*- coding:utf-8 -*-


import numpy as np
import mxnet as mx
from mxnet import nd
import cv2
import time

from generate_anchor import generate_anchors_fpn, nonlinear_pred
from numpy import frombuffer, uint8, concatenate, float32, block, maximum, minimum
from functools import partial


class BaseDetection:
    def __init__(self, *, thd, gpu, margin, nms_thd, verbose):
        self.threshold = thd
        self.nms_threshold = nms_thd
        self.device = gpu
        self.margin = margin

        self._nms_wrapper = partial(
            self.non_maximum_suppression, thresh=self.nms_threshold)

    def margin_clip(self, b):
        margin_x = (b[2] - b[0]) * self.margin
        margin_y = (b[3] - b[1]) * self.margin

        b[0] -= margin_x
        b[1] -= margin_y
        b[2] += margin_x
        b[3] += margin_y

        return np.clip(b, 0, None, out=b)

    @staticmethod
    def non_maximum_suppression(dets, thresh):
        """
        :param dets: [[x1, y1, x2, y2 score]]
        """
        x1, y1, x2, y2, scores = dets.T

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        while order.size > 0:
            keep, others = order[0], order[1:]

            yield dets[keep]

            xx1 = maximum(x1[keep], x1[others])
            yy1 = maximum(y1[keep], y1[others])
            xx2 = minimum(x2[keep], x2[others])
            yy2 = minimum(y2[keep], y2[others])

            w = maximum(0.0, xx2 - xx1 + 1)
            h = maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            ovr = inter / (areas[keep] - inter + areas[others])

            order = others[ovr < thresh]

    def non_maximum_selection(self, x):
        return x[:1]

    @staticmethod
    def filter_boxes(boxes, min_size, max_size=-1):
        """ Remove all boxes with any side smaller than min_size """
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        if max_size > 0:
            boxes = np.where(np.minimum(ws, hs) < max_size)[0]
        if min_size > 0:
            boxes = np.where(np.maximum(ws, hs) > min_size)[0]
        return boxes


class MxnetDetectionModel(BaseDetection):
    def __init__(self, prefix, epoch, scale, gpu=-1, thd=0.5, margin=0,
                 nms_thd=0.1, verbose=False):

        super().__init__(thd=thd, gpu=gpu, margin=margin, nms_thd=nms_thd, verbose=verbose)

        self.scale = scale
        self._rescale = partial(cv2.resize, dsize=None, fx=self.scale,
                                fy=self.scale, interpolation=cv2.INTER_LINEAR)

        self._ctx = mx.cpu() if self.device < 0 else mx.gpu(self.device)
        self._fpn_anchors = generate_anchors_fpn().items()
        self._runtime_anchors = {}

        model = self._load_model(prefix, epoch)

        self._forward = partial(model.forward, is_train=False)
        self._solotion = model.get_outputs

    def _load_model(self, prefix, epoch):
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        model = mx.mod.Module(sym, context=self._ctx, label_names=None)
        model.bind(data_shapes=[('data', (1, 3, 640, 480))],
                   for_training=False)
        model.set_params(arg_params, aux_params)
        return model

    def _anchors_plane(self, height, width, stride, base_anchors):
        """
        Parameters
        ----------
        height: height of plane
        width:  width of plane
        stride: stride ot the original image
        anchors_base: (A, 4) a base set of anchors

        Returns
        -------
        all_anchors: (height * width, A, 4) ndarray of anchors spreading over the plane
        """

        key = (height, width, stride)

        def gen_runtime_anchors():
            A = base_anchors.shape[0]

            all_anchors = np.zeros((height*width, A, 4), dtype=float32)

            rw = np.tile(np.arange(0, width*stride, stride), height)
            rh = np.repeat(np.arange(0, height*stride, stride), width)

            all_anchors += np.stack((rw, rh, rw, rh),
                                    axis=1).reshape(height*width, 1, 4)
            all_anchors += base_anchors

            self._runtime_anchors[key] = all_anchors

            return all_anchors

        return self._runtime_anchors[key] if key in self._runtime_anchors else gen_runtime_anchors()

    def _retina_detach(self, out, scale):
        out = map(lambda x: x.asnumpy(), out)

        def deal_with_fpn(fpn, scores, deltas):
            anchors = self._anchors_plane(
                *deltas.shape[-2:], *fpn).reshape((-1, 4))

            scores = scores[:, fpn[1].shape[0]:, :, :].transpose(
                (0, 2, 3, 1)).reshape((-1, 1))
            deltas = deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

            mask = scores.reshape((-1,)) > self.threshold
            proposals = deltas[mask]

            nonlinear_pred(anchors[mask], proposals)

            return [proposals / scale, scores[mask]]

        return block([deal_with_fpn(fpn, next(out), next(out)) for fpn in self._fpn_anchors])


    def workflow_inference(self):

        vc = cv2.VideoCapture(0)  # 读入视频文件

        while True:
            ret,frame=vc.read()
            st = time.time()

            dst = self._rescale(frame)
            data = nd.array([dst.transpose((2, 0, 1))])
            db = mx.io.DataBatch(data=(data,))
            self._forward(db)
            out= self._solotion()

            detach = self._retina_detach(out, self.scale)

            for res in self._nms_wrapper(detach):
                self.margin_clip(res)
                cv2.rectangle(frame, (res[0], res[1]),
                              (res[2], res[3]), (255, 255, 0))

            print(f'net: {time.time() - st}')
            cv2.imshow('res', frame)
            cv2.waitKey(1)


if __name__ == '__main__':
    width = 448
    scale=448/640
    fd = MxnetDetectionModel("weights/16and32", 0,scale=scale, gpu=-1, margin=0.15)
    fd.workflow_inference()

