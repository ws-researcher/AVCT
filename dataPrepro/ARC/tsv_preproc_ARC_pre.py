import copy
import os
import random
import sys

pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(pythonpath)
sys.path.insert(0, pythonpath)
import os.path as op
import json, yaml, code, io
import numpy as np
import pandas as pd
from utils.tsv_file_ops import tsv_writer
from utils.tsv_file_ops import generate_linelist_file
from collections import defaultdict

# data path to raw video files
data_vid_name = "/usb/ARC/videos/{}.mp4"
data_vid_id = "datasets/ARC/{}-{}-{}"

# data_vid_id = "/usb/YouCook2/videos/{}/{}"
dataset_path = './datasets/ARC/0/'
# annotations downloaded from official downstream dataset
anns = './datasets/ARC/annotation.json'

abnormal_train_keys_p = op.join(dataset_path, "abnormal_train_keys")
abnormal_val_keys_p = op.join(dataset_path, "abnormal_val_keys")

# To generate tsv files:
# {}.img.tsv: we use it to store video path info
visual_file = op.join(dataset_path, "{}.img.tsv")
# {}.caption.tsv: we use it to store  captions
cap_file = op.join(dataset_path, "{}.caption.tsv")
# {}.linelist.tsv: since each video may have multiple captions, we need to store the corresponance between vidoe id and caption id
linelist_file = op.join(dataset_path, "{}.linelist.tsv")
# {}.label.tsv: we store any useful labels or metadara here, such as object tags. Now we only have captions. maybe can remove it in future.
label_file = op.join(dataset_path, "{}.label.tsv")


def write_to_yaml_file(context, file_name):
    with open(file_name, 'w') as fp:
        yaml.dump(context, fp, encoding='utf-8')


def tsv_reader(tsv_file, sep='\t'):
    with open(tsv_file, 'r') as fp:
        for i, line in enumerate(fp):
            yield [x.strip() for x in line.split(sep)]


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def config_save_file(tsv_file, save_file=None, append_str='.new.tsv'):
    if save_file is not None:
        return save_file
    return op.splitext(tsv_file)[0] + append_str


def generate_caption_linelist_file(caption_tsv_file, save_file=None):
    captions = []
    for row in tsv_reader(caption_tsv_file):
        captions.append(len(json.loads(row[1])))
    num_captions = len(captions)
    cap_linelist = ['\t'.join([str(img_idx), str(cap_idx)])
                    for img_idx in range(num_captions)
                    for cap_idx in range(captions[img_idx])
                    ]

    save_file = config_save_file(caption_tsv_file, save_file, '.linelist.tsv')
    with open(save_file, 'w') as f:
        f.write('\n'.join(cap_linelist))
    return save_file


def dump_tsv_gt_to_coco_format(caption_tsv_file, outfile, cap_key="observed"):
    annotations = []
    images = []
    cap_id = 0
    caption_tsv = tsv_reader(caption_tsv_file)

    for cap_row in caption_tsv:
        image_id = cap_row[0]
        key = image_id
        caption_data = json.loads(cap_row[1])
        count = len(caption_data)
        for i in range(count):
            caption1 = caption_data[i][cap_key]
            annotations.append(
                {'image_id': image_id, 'caption': caption1,
                 'id': cap_id})
            cap_id += 1

        images.append({'id': image_id, 'file_name': key})

    with open(outfile, 'w') as fp:
        json.dump({'annotations': annotations, 'images': images,
                   'type': 'captions', 'info': 'dummy', 'licenses': 'dummy'},
                  fp)


def process_new(split):
    f = open(anns, 'r')
    database = json.load(f)

    video_list = set()
    for key in database.keys():
        video_name = key
        if os.path.exists(data_vid_name.format(video_name)):
            video_list.add(video_name)
    video_list = list(video_list)
    video_list.sort()

    img_label, img_label_abnormal, img_label_normal = [], [], []
    rows_label, rows_label_abnormal, rows_label_normal = [], [], []
    caption_label, caption_label_abnormal, caption_label_normal = [], [], []
    abnormal, normal = [], []

    for video_name in video_list:

        annotations = database[video_name]['annotations']
        annotations = [annotation for annotation in annotations if annotation.get("id") != "default"]
        annotations.sort(key=lambda x: x.get("id"))

        num_ann = len(annotations)

        # 去除非连续标注的视频
        if num_ann < 2:
            continue

        for i in range(num_ann - 1):
            segment_observed = annotations[i]
            segment_unobserved = annotations[i + 1]

            sample_observed_id = segment_observed['id']
            sample_unobserved_id = segment_unobserved['id']

            if sample_observed_id.split("-")[0] != sample_unobserved_id.split("-")[0]:
                continue

            timestamp_observed = segment_observed.get("segment")
            caption_observed = segment_observed.get("sentence")
            label_observed = segment_observed.get("is_accident")

            timestamp_unobserved = segment_unobserved.get("segment")
            caption_unobserved = segment_unobserved.get("sentence")

            start_time_observed = int(timestamp_observed[0] * 10)
            end_time_observed = int(timestamp_observed[1] * 10)

            start_time_unobserved = int(timestamp_unobserved[0] * 10)
            end_time_unobserved = int(timestamp_unobserved[1] * 10)

            if 1:
                # if start_time_unobserved > start_time_observed and end_time_unobserved > end_time_observed and start_time_unobserved == end_time_observed:
                # if start_time_unobserved > start_time_observed and end_time_unobserved > end_time_observed:
                # if start_time_unobserved >= end_time_observed:

                resolved_data_vid_id = data_vid_id.format(start_time_observed, end_time_observed, video_name)
                f_resolved_data_vid_id = data_vid_id.format(start_time_unobserved, end_time_unobserved, video_name)
                resolved_data_vid_name = data_vid_name.format(video_name)


                output_captions, output_labels = [], []
                output_captions.append({"caption_observed": caption_observed, "caption_unobserved": caption_unobserved,
                                        "caption_all": caption_observed + " " + caption_unobserved})
                output_labels.append({"label": int(label_observed)})

                if label_observed:
                    abnormal.append(([str(resolved_data_vid_id), json.dumps(output_captions), str(f_resolved_data_vid_id)],
                                     [str(resolved_data_vid_id), json.dumps(output_labels)],
                                     [str(resolved_data_vid_id), str(resolved_data_vid_name), start_time_observed, end_time_observed],
                                     [str(f_resolved_data_vid_id), str(resolved_data_vid_name), start_time_unobserved, end_time_unobserved]))
                else:
                    normal.append(([str(resolved_data_vid_id), json.dumps(output_captions), str(f_resolved_data_vid_id)],
                                   [str(resolved_data_vid_id), json.dumps(output_labels)],
                                   [str(resolved_data_vid_id), str(resolved_data_vid_name), start_time_observed, end_time_observed],
                                   [str(f_resolved_data_vid_id), str(resolved_data_vid_name), start_time_unobserved, end_time_unobserved]))

    normal.extend(abnormal)

    num_all = int(len(normal))
    num_train = int(num_all * 0.7)
    num_val = num_all - num_train


    random.seed(0)  # 必须排序或者指定种子，不然会造成训练集和验证机重复
    random.shuffle(normal)

    if split == "train":
        train_samples = normal[:num_train]
        caption_label, rows_label, img_label_c, img_label_f = [item[0] for item in train_samples], \
                                                                          [item[1] for item in train_samples], \
                                                                          [item[2] for item in train_samples], \
                                                                          [item[3] for item in train_samples]

    elif split == "val":
        val_samples = normal[num_train:]
        caption_label, rows_label, img_label_c, img_label_f = [item[0] for item in val_samples], \
                                                                          [item[1] for item in val_samples], \
                                                                          [item[2] for item in val_samples], \
                                                                          [item[3] for item in val_samples]
    img_label = []
    img_label.extend(img_label_c)
    img_label.extend(img_label_f)
    resolved_visual_file = visual_file.format(split)
    print("generating visual file for", resolved_visual_file)
    tsv_writer(img_label, resolved_visual_file)

    resolved_label_file = label_file.format(split)
    print("generating label file for", resolved_label_file)
    tsv_writer(rows_label, resolved_label_file)

    resolved_linelist_file = linelist_file.format(split)
    print("generating linelist file for", resolved_linelist_file)
    generate_linelist_file(resolved_label_file, save_file=resolved_linelist_file)

    resolved_cap_file = cap_file.format(split)
    print("generating cap file for", resolved_cap_file)
    tsv_writer(caption_label, resolved_cap_file)
    print("generating cap linelist file for", resolved_cap_file)
    resolved_cap_linelist_file = generate_caption_linelist_file(resolved_cap_file)

    gt_file_coco = op.splitext(resolved_cap_file)[0] + '_observed_coco_format.json'
    print("convert gt to", gt_file_coco)
    dump_tsv_gt_to_coco_format(resolved_cap_file, gt_file_coco, cap_key="caption_observed")
    gt_file_coco = op.splitext(resolved_cap_file)[0] + '_unobserved_coco_format.json'
    print("convert gt to", gt_file_coco)
    dump_tsv_gt_to_coco_format(resolved_cap_file, gt_file_coco, cap_key="caption_unobserved")

    out_cfg = {}
    all_field = ['img', 'label', 'caption', 'caption_linelist', 'caption_coco_format']
    all_tsvfile = [resolved_visual_file, resolved_label_file, resolved_cap_file, resolved_cap_linelist_file,
                   gt_file_coco]
    for field, tsvpath in zip(all_field, all_tsvfile):
        out_cfg[field] = tsvpath.split('/')[-1]
    out_yaml = '{}.yaml'.format(split)
    write_to_yaml_file(out_cfg, op.join(dataset_path, out_yaml))
    print('Create yaml file: {}'.format(op.join(dataset_path, out_yaml)))


def main():
    process_new('train')
    process_new('val')


if __name__ == '__main__':
    main()



