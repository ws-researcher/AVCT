import argparse
import json
import yaml
from easydict import EasyDict
from learn.MVVC import MVVC
from utils.comm import dist_init
from utils.miscellaneous import check_yaml_file, set_seed
import os.path as op

def basic_check_arguments(args):
    args.output_dir = args.output_dir.replace(" ", "_")

    # can add some basic checks here
    if args.mixed_precision_method != "deepspeed":
        args.zero_opt_stage = -1
        args.deepspeed_fp16 = False

    if args.mixed_precision_method != "fairscale":
        args.zero_opt_stage = -1
        args.fairscale_fp16 = False

    if args.mixed_precision_method != "apex":
        args.restore_ratio = -1

    if args.text_mask_type != "pos_tag":
        args.mask_tag_prob = -1

    if hasattr(args, 'do_train') and args.do_train:
        check_yaml_file(op.join(args.data_dir, args.train_yaml))
        if args.evaluate_during_training:
            check_yaml_file(op.join(args.data_dir, args.val_yaml))

        assert args.per_gpu_train_batch_size > 0
        args.effective_batch_size = args.per_gpu_train_batch_size * args.num_gpus
        args.per_gpu_eval_batch_size = max(
            args.per_gpu_eval_batch_size, args.per_gpu_train_batch_size)

        if args.predict_caption == 2:
            assert args.max_seq_length == 2 * args.max_seq_a_length
        else:
            assert args.max_seq_length == args.max_seq_a_length

        # if args.use_asr:
        #     args.add_od_labels = True
        # if args.add_od_labels:
        #     assert args.max_seq_length > args.max_seq_a_length
        # elif args.use_sep_cap:
        #     assert args.max_seq_length == 2 * args.max_seq_a_length
        # else:
        #     assert args.max_seq_length == args.max_seq_a_length
        # if args.use_swap_cap:
        #     assert args.use_sep_cap
    if hasattr(args, 'do_test') and args.do_test:
        for test_yaml in args.test_yaml:
            check_yaml_file(op.join(args.data_dir, test_yaml))

def check_arguments(args):
    # shared basic checks
    basic_check_arguments(args)
    # additional sanity check:
    args.max_img_seq_length = int((args.max_num_frames / 2) * (int(args.img_res) / 32) * (int(args.img_res) / 32))

    if args.freeze_backbone or args.backbone_coef_lr == 0:
        args.backbone_coef_lr = 0
        args.freeze_backbone = True

    if 'reload_pretrained_swin' not in args.keys():
        args.reload_pretrained_swin = False

    if not len(args.pretrained_checkpoint) and args.reload_pretrained_swin:
        args.reload_pretrained_swin = False

    if args.learn_mask_enabled == True and args.attn_mask_type != 'learn_without_crossattn' and args.attn_mask_type != 'learn_with_swap_crossattn':
        args.attn_mask_type = 'learn_vid_att'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="path to the config file",
                        default="learn/configs/config_ARC.yml") # config_BDDX  config_MSRVTT   config_YouCook2   config_ActivityNetCaption
    parser.add_argument("--local_rank", type=int, default=0,
                        help="For distributed training.")

    args = parser.parse_args()

    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)
        class obj(object):
            def __init__(self, dict_):
                self.__dict__.update(dict_)
        config =  json.loads(json.dumps(config), object_hook=obj)
        config = EasyDict(vars(config))
        dist_init(config)
        check_arguments(config)
        set_seed(config.seed, config.num_gpus)


        model = MVVC(config)
        model.train()