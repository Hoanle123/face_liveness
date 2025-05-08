import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms
from torchkit.backbone import get_model as get_tface_model
import torchvision.transforms as transforms, models


class RetinaFaceInfer:
    def __init__(self, weight_path="weights/mobilenet0.25_Final.pth", confidence_threshold=0.95, vis_thres=0.6) -> None:
        self.cfg = None
        self.confidence_threshold = confidence_threshold
        self.vis_thres = vis_thres
        if "mobilenet" in weight_path.lower():
            self.cfg = cfg_mnet
        elif "resnet50" in weight_path.lower():
            self.cfg = cfg_re50
        self.net = RetinaFace(cfg=self.cfg, phase='test')
        self.net = self.load_model(self.net, weight_path, load_to_cpu=True)
        self.net.eval()
        print('Finished loading model!')

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.net = self.net.to(self.device)
        self.resize = 1

    @staticmethod
    def check_keys(model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print('Missing keys:{}'.format(len(missing_keys)))
        print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(
            used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True

    @staticmethod
    def remove_prefix(state_dict, prefix):
        """ Old style model is stored with all names of parameters sharing common prefix 'module.' """
        print('remove prefix \'{}\''.format(prefix))

        def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x

        return {f(key): value for key, value in state_dict.items()}

    def load_model(self, model, pretrained_path, load_to_cpu):
        print('Loading pretrained model from {}'.format(pretrained_path))
        if load_to_cpu:
            pretrained_dict = torch.load(
                pretrained_path, map_location=lambda storage, loc: storage)
        else:
            self.device = torch.cuda.current_device()
            pretrained_dict = torch.load(
                pretrained_path, map_location=lambda storage, loc: storage.cuda(self.device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(
                pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        self.check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model

    def detect(self, img_raw="./curve/test.jpg", show=False, nms_threshold=0.4, top_k=5000, keep_top_k=750,
               save_image=False):
        if isinstance(img_raw, str):
            img_raw = cv2.imread(img_raw, cv2.IMREAD_COLOR)
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor(
            [img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        loc, conf, landms = self.net(img)  # forward pass

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / self.resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(
            0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / self.resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
            np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)

        # show image
        if show:
            for b in dets:
                if b[4] < self.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]),
                              (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            # save image
            plt.imshow(img_raw)
            plt.show()
        return dets, img_raw


def __get_rotated_angle__(point_1, point_2):
    """
    Returns the angle in degrees between two points.
    """
    x_diff = point_2[0] - point_1[0]
    y_diff = point_2[1] - point_1[1]
    angle = np.rad2deg(np.arctan2(y_diff, x_diff))
    return angle


class FaceAlignment:

    def __init__(self):
        pass

    @staticmethod
    def __get_rotation_matrix__(center, angle):
        """
        Returns the rotation matrix for the given angle.
        """
        return cv2.getRotationMatrix2D(center, angle, 1)

    def __image_rotation__(self, image, angle):
        """
        Rotates an image by the given angle.
        """
        rows, cols = image.shape[:2]
        M = self.__get_rotation_matrix__((cols / 2, rows / 2), angle)
        return cv2.warpAffine(image, M, (cols, rows))

    @staticmethod
    def __get_rotated_bounding_box__(image, bounding_box, angle):
        """
        Returns a rotated bounding box.
        """
        cen_x, cen_y = image.shape[1] / 2, image.shape[0] / 2

        l, t, r, b = bounding_box
        # r, b = l + w, t + h
        w, h = r - l, b - t

        # positional = np.array([[l, t], [r, t], [l, b], [r, b]])
        # M = self.__get_rotation_matrix__((cen_x, cen_y), angle)
        # print('positional: ', positional,'M: ', M, 'W,H: ', w, h)
        # rotated = cv2.warpAffine(positional, M, (w, h))
        # return rotated.astype(int)
        # Tạo ma trận xoay
        rotation_matrix = cv2.getRotationMatrix2D((cen_x, cen_y), angle, 1)

        # Tạo mảng chứa tọa độ của các điểm sau khi xoay
        rotated_points = np.array([[l, t], [r, t], [l, b], [r, b]], dtype=np.float32)

        # Áp dụng ma trận xoay lên các điểm của bounding box
        rotated_points = cv2.transform(rotated_points.reshape(1, -1, 2), rotation_matrix).reshape(-1, 2)

        # Tính toán bounding box mới
        new_l = max(np.min(rotated_points[:, 0]), 0)
        new_t = max(np.min(rotated_points[:, 1]), 0)
        new_r = min(np.max(rotated_points[:, 0]), image.shape[1])
        new_b = min(np.max(rotated_points[:, 1]), image.shape[0])

        return np.array([new_l, new_t, new_r, new_b]).astype(int)

    def align(self, image, det):
        det = list(map(int, det))
        left_eye = (det[5], det[6])
        right_eye = (det[7], det[8])

        # Get the angle between the eyes
        angle = __get_rotated_angle__(left_eye, right_eye)
        # print('angle: ', angle)
        M = self.__get_rotation_matrix__((image.shape[1] / 2, image.shape[0] / 2), angle)
        bounding_box = np.array(det[:4])
        # print('bounding_box ', bounding_box)
        return self.__image_rotation__(image, angle), self.__get_rotated_bounding_box__(image, bounding_box, angle)

    @staticmethod
    def straight(det):
        det = list(map(int, det))
        left_eye = (det[5], det[6])
        right_eye = (det[7], det[8])
        nose = (det[9], det[10])
        left_0 = (det[5], 0)
        right_0 = (det[7], 0)
        if left_eye[0] < nose[0] < right_eye[0]:

            line1 = np.array([left_eye, left_0], dtype=np.float32)
            distance1 = cv2.pointPolygonTest(line1, nose, True)

            line2 = np.array([right_eye, right_0], dtype=np.float32)
            distance2 = cv2.pointPolygonTest(line2, nose, True)
            # print('distance1', distance1, distance2, distance1 / distance2)
            return distance1 / distance2
        else:
            return 0

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    facemodel = RetinaFaceInfer()
    dets = facemodel.detect()
    print(dets)
