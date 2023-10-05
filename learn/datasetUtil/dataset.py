import json

from PIL import Image

from learn.datasetUtil.image_ops import img_from_base64
from learn.datasetUtil.video_transforms import *
from learn.datasetUtil.volume_transforms import ClipToTensor
from utils.load_files import load_from_yaml_file, find_file_path_in_yaml, load_box_linelist_file
import os.path as op
import numpy as np

from utils.tsv_file import TSVFile


class VisionLanguageTSVDataset(object):
    def __init__(self, args, tokenizer_multimodal, is_train):
        self.args = args
        self.tokenizer_multimodal = tokenizer_multimodal
        self.is_train = is_train

        self.readData()
        self.raw_video_prcoess = self.video_prcoess()

    def readData(self):
        if self.is_train:
            annotionFile = op.join(self.args.data_dir, self.args.train_yaml)
        else:
            annotionFile = op.join(self.args.data_dir, self.args.val_yaml)

        cfg = load_from_yaml_file(annotionFile)
        self.root = op.dirname(annotionFile)
        cap_linelist_file = find_file_path_in_yaml(cfg.get('caption_linelist', None), self.root)

        # line_list = load_box_linelist_file(cap_linelist_file)
        line_list = np.array(load_box_linelist_file(cap_linelist_file))
        self.img_line_list_c = line_list[0]
        self.cap_line_list = line_list[1]
        self.img_line_list_f = line_list[0] + len(self.img_line_list_c)

        self.visual_file = cfg.get('img', None)
        self.visual_tsv = self.get_tsv_file(self.visual_file)

        self.cap_file = cfg.get('caption', None)
        self.cap_tsv = self.get_tsv_file(self.cap_file)

        self.image_keys = [self.cap_tsv[i][0] for i in range(self.cap_tsv.num_rows())]
        self.image_keys.extend([self.cap_tsv[i][2] for i in range(self.cap_tsv.num_rows())])
        print("s")

    def video_prcoess(self):
        img_res = getattr(self.args, 'img_res', 224)

        if self.is_train:
            self.raw_video_crop_list = [
                Resize(img_res),
                RandomCrop((img_res,img_res)),
                ClipToTensor(channel_nb=3),
                Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ]
        else:
            self.raw_video_crop_list = [
                Resize(img_res),
                CenterCrop((img_res, img_res)),
                ClipToTensor(channel_nb=3),
                Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ]
        raw_video_prcoess = Compose(self.raw_video_crop_list)
        return raw_video_prcoess

    def get_tsv_file(self, tsv_file):
        if tsv_file:
            tsv_path = find_file_path_in_yaml(tsv_file, self.root)
            return TSVFile(tsv_path)

    def get_row_from_tsv(self, tsv, img_idx):
        row = tsv[img_idx]
        assert row[0].split('/')[0] == self.image_keys[img_idx].split('_')[-1] or \
               row[0].split('_')[-1] == self.image_keys[img_idx].split('_')[-1]
        return row

    def get_caption(self, img_idx, cap_idx):
        row = self.cap_tsv[img_idx]
        assert row[0].split('/')[0] == self.image_keys[img_idx].split('_')[-1] or \
               row[0].split('_')[-1] == self.image_keys[img_idx].split('_')[-1]

        caption = json.loads(row[1])[cap_idx]
        return caption

    def get_visual_data(self, idx):
        frames = []
        row = self.get_row_from_tsv(self.visual_tsv, idx)[2:]
        for i in range(len(row)):
            cv2_im = img_from_base64(row[i])
            cv2_im = cv2_im[:, :, ::-1]  # COLOR_BGR2RGB
            image = np.transpose(cv2_im[np.newaxis, ...], (0, 3, 1, 2))
            frames.append(image)
        return np.vstack(frames)

    def apply_augmentations(self, frames):

        frames = frames.astype(np.uint8)
        frames = np.transpose(frames, (0, 2, 3, 1))

        frame_list = [Image.fromarray(frame) for frame in frames]

        # adapt from torch_videovision: https://github.com/hassony2/torch_videovision
        # after augmentation, output tensor (C x T x H x W) in the range [0, 1.0]
        crop_frames = self.raw_video_prcoess(frame_list)
        # (C x T x H x W) --> (T x C x H x W)
        crop_frames = crop_frames.permute(1, 0, 2, 3)
        return crop_frames

    def __len__(self):
        return len(self.img_line_list_c)

    def __getitem__(self, idx):
        img_idx_c = self.img_line_list_c[idx]
        img_idx_f = self.img_line_list_f[idx]
        cap_idx = self.cap_line_list[idx]

        img_key_c = self.image_keys[img_idx_c]
        # img_key_f = self.image_keys_f[img_idx_f]

        caption = self.get_caption(img_idx_c, cap_idx)
        raw_frames_c= self.get_visual_data(img_idx_c)
        raw_frames_f = self.get_visual_data(img_idx_f)
        preproc_frames_c = self.apply_augmentations(raw_frames_c)
        preproc_frames_f = self.apply_augmentations(raw_frames_f)

        # preproc_frames = torch.cat((preproc_frames_f, preproc_frames_c), 0)
        preproc_frames = preproc_frames_c

        example1, example2 = None, None
        if self.args.predict_caption == 0:
            example = self.tokenizer_multimodal.tensorize_example_VTT(caption["caption_observed"], preproc_frames)
        elif self.args.predict_caption == 1:
            example = self.tokenizer_multimodal.tensorize_example_VTT(caption["caption_unobserved"], preproc_frames)
        elif self.args.predict_caption == 2:
            example1 = self.tokenizer_multimodal.tensorize_example_VTT(caption["caption_observed"], preproc_frames, text_b=caption["caption_unobserved"])
            # example1 = self.tokenizer_multimodal.tensorize_example_VTT(caption["caption_unobserved"], preproc_frames, text_b=caption["caption_observed"])

            # example2 = self.tokenizer_multimodal.tensorize_example_VVT(caption["caption_unobserved"], preproc_frames, img_feat_f=preproc_frames_f)

            example = self.tokenizer_multimodal.tensorize_example_VVTT(caption["caption_observed"], preproc_frames, text_b=caption["caption_unobserved"], img_feat_f=preproc_frames_f)


        # preparing outputs
        meta_data = {}
        meta_data['caption'] = caption # raw text data, not tokenized
        meta_data['img_key'] = img_key_c
        # meta_data['is_video'] = is_video # True: video data, False: image data
        # meta_data['tag'] = tag
        return img_key_c, example, meta_data

        # return img_key_c, example1, meta_data