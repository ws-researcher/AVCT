import torch
from fairscale.nn import checkpoint_wrapper

from model.clip.model import Transformer
from model.extracter import Extracter
from model.video_bert import BertConfig, BertForImageCaptioning, BertTokenizer
from model.clip import clip
from model.video_swin.config import Config
from model.video_swin.swin_transformer import SwinTransformer3D


class VideoTransformer(torch.nn.Module):
    def __init__(self, args):
        super(VideoTransformer, self).__init__()

        self.extracter = Extracter(args)

        # self.Transformer = Transformer(int(args.img_feature_dim), 2, 8)

        transformer_encoder, config, self.tokenizer = get_bert_model(args)
        self.config = config
        self.trans_encoder = transformer_encoder

        self.compute_mask_on_the_fly = False  # deprecated
        self.mask_prob = args.mask_prob
        self.mask_token_id = -1
        self.max_img_seq_length = args.max_num_frames

        self.max_num_frames = getattr(args, 'max_num_frames', 2)

        # learn soft attention mask
        self.learn_mask_enabled = getattr(args, 'learn_mask_enabled', False)
        self.sparse_mask_soft2hard = getattr(args, 'sparse_mask_soft2hard', False)

        if self.learn_mask_enabled == True:
            self.learn_vid_att = torch.nn.Embedding(self.max_img_seq_length  * self.max_img_seq_length , 1)
            self.sigmoid = torch.nn.Sigmoid()

    def forward(self, *args, **kwargs):
        images = kwargs['img_feats']
        images_f = kwargs['img_feats_f']

        vid_feats = self.extracter(images)
        vid_feats_f = self.extracter(images_f)

        # vid_feats = self.Transformer(vid_feats)

        kwargs['img_feats'] = vid_feats
        kwargs['img_feats_f'] = vid_feats_f

        # kwargs['img_feats'] = torch.cat((vid_feats, vid_feats_f), 1)

        # prepare VL transformer inputs

        if self.trans_encoder.bert.encoder.output_attentions:
            self.trans_encoder.bert.encoder.set_output_attentions(False)
        # learn soft attention mask
        if self.learn_mask_enabled:
            kwargs['attention_mask'] = kwargs['attention_mask'].float()
            vid_att_len = self.max_img_seq_length
            learn_att = self.learn_vid_att.weight.reshape(vid_att_len, vid_att_len)
            learn_att = self.sigmoid(learn_att)
            diag_mask = torch.diag(torch.ones(vid_att_len)).cuda()
            video_attention = (1. - diag_mask) * learn_att
            learn_att = diag_mask + video_attention
            if self.sparse_mask_soft2hard:
                learn_att = (learn_att >= 0.5) * 1.0
                learn_att = learn_att.cuda()
                learn_att.requires_grad = False
            kwargs['attention_mask'][:, -vid_att_len::, -vid_att_len::] = learn_att
        outputs = self.trans_encoder(*args, **kwargs)
        if self.learn_mask_enabled:
            loss_sparsity = self.get_loss_sparsity(video_attention)
            outputs = outputs + (loss_sparsity,)
        return outputs

    def get_loss_sparsity(self, video_attention):
        sparsity_loss = 0
        sparsity_loss += (torch.mean(torch.abs(video_attention)))
        return sparsity_loss

    def diag_based_init_attn_mask(self, pretrain_attn_mask):
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
            pretrained_num_tokens, pretrained_num_tokens)
        zeros_mask = torch.zeros_like(pretrained_learn_att)
        scale_factor = self.max_img_seq_length / pretrained_num_tokens

        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len, vid_att_len)
        with torch.no_grad():
            for i in range(int(scale_factor)):
                learn_att[pretrained_num_tokens * i:pretrained_num_tokens * (i + 1),
                pretrained_num_tokens * i:pretrained_num_tokens * (i + 1)] = pretrained_learn_att

    def bilinear_init_attn_mask(self, pretrain_attn_mask):
        print('init attn mask with bilinear interpolation')
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
            pretrained_num_tokens, pretrained_num_tokens)
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len, vid_att_len)
        scale_factor = int(self.max_img_seq_length / pretrained_num_tokens)
        sampler = torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        with torch.no_grad():
            learn_att = sampler(pretrained_learn_att[None, None, :, :].double())[0, 0, :, :].half()

    def random_init_attn_mask(self):
        print('random init attn mask')
        self.learn_vid_att = torch.nn.Embedding(self.max_img_seq_length * self.max_img_seq_length, 1)

    def reload_attn_mask(self, pretrain_attn_mask):
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
            pretrained_num_tokens, pretrained_num_tokens)
        scale_factor = 1
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len, vid_att_len)
        with torch.no_grad():
            for i in range(int(scale_factor)):
                learn_att[pretrained_num_tokens * i:pretrained_num_tokens * (i + 1),
                pretrained_num_tokens * i:pretrained_num_tokens * (i + 1)] = pretrained_learn_att

    def freeze_backbone(self, freeze=True):
        for _, p in self.extracter.Clip_model.named_parameters():
            p.requires_grad = not freeze

class myVideoSwin(torch.nn.Module):
    def __init__(self, args, cfg, backbone):
        super(myVideoSwin, self).__init__()
        self.backbone = backbone
        self.use_grid_feature = args.grid_feat

    def forward(self, x):
        x = self.backbone(x)
        return x

def get_clip_model(args):
    model, preprocess = clip.load('ViT-B/32')

    # model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e31')

    return model

def get_swin_model(args):
    if int(args.img_res) == 384:
        assert args.vidswin_size == "large"
        config_path = 'model/video_swin/swin_%s_384_patch244_window81212_kinetics%s_22k.py'%(args.vidswin_size, args.kinetics)
        model_path = './models/video_swin_transformer/swin_%s_384_patch244_window81212_kinetics%s_22k.pth'%(args.vidswin_size, args.kinetics)
    else:
        # in the case that args.img_res == '224'
        config_path = 'model/video_swin/swin_%s_patch244_window877_kinetics%s_22k.py'%(args.vidswin_size, args.kinetics)
        model_path = './models/video_swin_transformer/swin_%s_patch244_window877_kinetics%s_22k.pth'%(args.vidswin_size, args.kinetics)
    if args.pretrained_2d:
        config_path = 'src/modeling/video_swin/swin_base_patch244_window877_kinetics400_22k.py'
        model_path = './models/swin_transformer/swin_base_patch4_window7_224_22k.pth'

    cfg = Config.fromfile(config_path)
    pretrained_path = model_path if args.pretrained_2d else None
    backbone = SwinTransformer3D(
                    pretrained=pretrained_path,
                    pretrained2d=args.pretrained_2d,
                    patch_size=cfg.model['backbone']['patch_size'],
                    in_chans=3,
                    embed_dim=cfg.model['backbone']['embed_dim'],
                    depths=cfg.model['backbone']['depths'],
                    num_heads=cfg.model['backbone']['num_heads'],
                    window_size=cfg.model['backbone']['window_size'],
                    mlp_ratio=4.,
                    qkv_bias=True,
                    qk_scale=None,
                    drop_rate=0.,
                    attn_drop_rate=0.,
                    drop_path_rate=0.2,
                    norm_layer=torch.nn.LayerNorm,
                    patch_norm=cfg.model['backbone']['patch_norm'],
                    frozen_stages=-1,
                    use_checkpoint=False)

    video_swin = myVideoSwin(args=args, cfg=cfg, backbone=backbone)

    if not args.pretrained_2d:
        checkpoint_3d = torch.load(model_path, map_location='cpu')
        video_swin.load_state_dict(checkpoint_3d['state_dict'], strict=False)
    else:
        video_swin.backbone.init_weights()
    return video_swin

def get_bert_model(args):
    # Load pretrained bert and tokenizer based on training configs
    config_class, model_class, tokenizer_class = BertConfig, BertForImageCaptioning, BertTokenizer
    config = config_class.from_pretrained(args.config_name if args.config_name else \
            args.model_name_or_path, num_labels=2, finetuning_task='image_captioning')

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name \
            else args.model_name_or_path, do_lower_case=args.do_lower_case)
    config.img_feature_type = 'frcnn'
    config.hidden_dropout_prob = args.drop_out
    config.loss_type = 'classification'
    config.tie_weights = args.tie_weights
    config.freeze_embedding = args.freeze_embedding
    config.label_smoothing = args.label_smoothing
    config.drop_worst_ratio = args.drop_worst_ratio
    config.drop_worst_after = args.drop_worst_after

    config.t = args.t
    config.lamda = args.lamda
    # update model structure if specified in arguments
    update_params = ['img_feature_dim', 'num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']
    model_structure_changed = [False] * len(update_params)
    # model_structure_changed[0] = True  # cclin hack
    for idx, param in enumerate(update_params):
        arg_param = getattr(args, param)
        # bert-base-uncased do not have img_feature_dim
        config_param = getattr(config, param) if hasattr(config, param) else -1
        if arg_param > 0 and arg_param != config_param:
            setattr(config, param, arg_param)
            model_structure_changed[idx] = True
    if any(model_structure_changed):
        assert config.hidden_size % config.num_attention_heads == 0
        if args.load_partial_weights:
            # can load partial weights when changing layer only.
            assert not any(model_structure_changed[2:]), "Cannot load partial weights " \
                "when any of ({}) is changed.".format(', '.join(update_params[2:]))
            model = model_class.from_pretrained(args.model_name_or_path,
                from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
        else:
            model = model_class(config=config) # init from scratch
    else:
        model = model_class.from_pretrained(args.model_name_or_path,
            from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    total_params = sum(p.numel() for p in model.parameters())
    return model, config, tokenizer