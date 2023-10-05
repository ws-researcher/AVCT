import os
import sys

pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(pythonpath)
sys.path.insert(0, pythonpath)

import os.path as op
import json, yaml
from utils.tsv_file_ops import tsv_writer, generate_linelist_file

# data path to raw video files
data_vid_name = "/usb/ActivityNetDataset/videos/{}"
data_vid_id = "datasets/ActivityNetCaption/{}_{}-{}-{}"

dataset_path = './datasets/ActivityNetCaption/'
# annotations downloaded from official downstream dataset
train_anns = 'datasets/ActivityNetCaption/train.json'
val_anns = 'datasets/ActivityNetCaption/val_1.json'

# To generate tsv files:
# {}.img.tsv: we use it to store video path info 
visual_file = "./datasets/ActivityNetCaption/{}.img.tsv"
# {}.caption.tsv: we use it to store  captions
cap_file = "./datasets/ActivityNetCaption/{}.caption.tsv"
# {}.linelist.tsv: since each video may have multiple captions, we need to store the corresponance between vidoe id and caption id
linelist_file = "./datasets/ActivityNetCaption/{}.linelist.tsv"
# {}.label.tsv: we store any useful labels or metadara here, such as object tags. Now we only have captions. maybe can remove it in future.
# label_file = "./datasets/ActivityNetCaption/{}.label.tsv"

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
    num_captions = []
    for row in tsv_reader(caption_tsv_file):
        num_captions.append(len(json.loads(row[1])))

    cap_linelist = ['\t'.join([str(img_idx), str(cap_idx)]) 
            for img_idx in range(len(num_captions)) 
            for cap_idx in range(num_captions[img_idx])
    ]
    save_file = config_save_file(caption_tsv_file, save_file, '.linelist.tsv')
    with open(save_file, 'w') as f:
        f.write('\n'.join(cap_linelist))
    return save_file

def dump_tsv_gt_to_coco_format(caption_tsv_file, outfile):
    annotations = []
    images = []
    cap_id = 0
    caption_tsv = tsv_reader(caption_tsv_file)

    for cap_row  in caption_tsv:
        image_id = cap_row[0]
        key = image_id
        caption_data = json.loads(cap_row[1])
        count = len(caption_data)
        for i in range(count):
            caption = caption_data[i]['caption']
            caption_unobserved = caption_data[i]['caption_unobserved']
            caption_all = caption_data[i]['caption_all']
            annotations.append(
                        {'image_id': image_id, 'caption': caption, "caption_unobserved": caption_unobserved, "caption_all": caption_all,
                        'id': cap_id})
            cap_id += 1

        images.append({'id': image_id, 'file_name': key})

    with open(outfile, 'w') as fp:
        json.dump({'annotations': annotations, 'images': images,
                'type': 'captions', 'info': 'dummy', 'licenses': 'dummy'},
                fp)

def process_new(split):
    if split == "train":
        f = open(train_anns, 'r')
    else:
        f = open(val_anns, 'r')
    annos = json.load(f)

    img_label = []
    rows_label = []
    caption_label = []

    for key in annos:
        video_name = key
        annotations = annos[key]
        timestamps = annotations["timestamps"]
        sentences = annotations["sentences"]

        num_ann = len(sentences)

        # 去除非连续标注的视频
        if num_ann < 2:
            continue

        for i in range(num_ann - 1):
            timestamp_observed = timestamps[i]
            caption_observed = sentences[i]

            timestamp_unobserved = timestamps[i + 1]
            caption_unobserved = sentences[i + 1]

            start_time_observed = int(timestamp_observed[0] * 100)
            end_time_observed = int(timestamp_observed[1] * 100)

            start_time_unobserved = int(timestamp_unobserved[0] * 100)
            end_time_unobserved = int(timestamp_unobserved[1] * 100)

            # 去除时间不衔接的注释
            if start_time_unobserved > start_time_observed and end_time_unobserved > end_time_observed and start_time_unobserved == end_time_observed:
            # if start_time_unobserved > start_time_observed and end_time_unobserved > end_time_observed:
            # if start_time_unobserved >= end_time_observed:

                resolved_data_vid_id = data_vid_id.format(split, start_time_observed, end_time_observed, video_name)
                resolved_data_vid_name = data_vid_name.format(video_name)

                if os.path.exists(resolved_data_vid_name + ".mp4"):
                    resolved_data_vid_name = resolved_data_vid_name + ".mp4"
                elif os.path.exists(resolved_data_vid_name + ".mkv"):
                    resolved_data_vid_name = resolved_data_vid_name + ".mkv"
                else:
                    print("s")

                output_captions = []
                output_captions.append({"caption": caption_observed, "caption_unobserved": caption_unobserved,
                                        "caption_all": caption_observed + " " + caption_unobserved})

                caption_label.append([str(resolved_data_vid_id), json.dumps(output_captions)])
                # rows_label.append([str(resolved_data_vid_id), json.dumps(output_captions)])
                img_label.append([str(resolved_data_vid_id), str(resolved_data_vid_name), start_time_observed, end_time_observed])

    resolved_visual_file = visual_file.format(split)
    print("generating visual file for", resolved_visual_file)
    tsv_writer(img_label, resolved_visual_file)

    # resolved_label_file = label_file.format(split)
    # print("generating label file for", resolved_label_file)
    # tsv_writer(rows_label, resolved_label_file)

    # resolved_linelist_file = linelist_file.format(split)
    # print("generating linelist file for", resolved_linelist_file)
    # generate_linelist_file(resolved_label_file, save_file=resolved_linelist_file)

    resolved_cap_file = cap_file.format(split)
    print("generating cap file for", resolved_cap_file)
    tsv_writer(caption_label, resolved_cap_file)
    print("generating cap linelist file for", resolved_cap_file)
    resolved_cap_linelist_file = generate_caption_linelist_file(resolved_cap_file)

    gt_file_coco = op.splitext(resolved_cap_file)[0] + '_coco_format.json'
    print("convert gt to", gt_file_coco)
    dump_tsv_gt_to_coco_format(resolved_cap_file, gt_file_coco)

    out_cfg = {}
    # all_field = ['img', 'label', 'caption', 'caption_linelist', 'caption_coco_format']
    # all_tsvfile = [resolved_visual_file, resolved_label_file, resolved_cap_file, resolved_cap_linelist_file, gt_file_coco]
    all_field = ['img', 'caption', 'caption_linelist', 'caption_coco_format']
    all_tsvfile = [resolved_visual_file, resolved_cap_file, resolved_cap_linelist_file, gt_file_coco]
    for field, tsvpath in zip(all_field, all_tsvfile):
        out_cfg[field] = tsvpath.split('/')[-1]
    out_yaml = '{}.yaml'.format(split)
    write_to_yaml_file(out_cfg, op.join(dataset_path, out_yaml))
    print('Create yaml file: {}'.format(op.join(dataset_path, out_yaml)))


def main():
    process_new('train')
    process_new('val')
    # process_new('test')

if __name__ == '__main__':
    main()



