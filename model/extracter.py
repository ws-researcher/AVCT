import torch
from fairscale.nn import checkpoint_wrapper

from model.clip import clip
from model.video_swin.config import Config
from model.video_swin.swin_transformer import SwinTransformer3D


class Extracter(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.use_grid_feat = args.grid_feat
        Clip_model = get_clip_model(args)

        self.Clip_model = Clip_model


    def forward(self, images):
        B, S, C, H, W = images.shape  # batch, segment, chanel, hight, width
        # (B x S x C x H x W) --> (B x C x S x H x W)
        images = images.permute(0, 2, 1, 3, 4)

        images = images.permute(0, 2, 1, 3, 4)
        images = images.view(-1, C, H, W)
        vid_app_feats = self.Clip_model.encode_image(images)
        vid_app_feats = vid_app_feats.view(B, S, -1)

        return vid_app_feats



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