import json

import deepspeed
import torch
from tqdm import tqdm

from evalcap.coco_caption.pycocoevalcap.eval import COCOEvalCap
from evalcap.coco_caption.pycocotools.coco import COCO
from evalcap.utils_caption_evaluate import convert_tsv_to_coco_format
from learn.datasetUtil.vl_dataloader import make_data_sampler, make_batch_data_sampler, init_seeds, make_data_loader
from learn.lr_scheduler import WarmupLinearLR
from learn.optimization import AdamW
from model.VLtransformer import VideoTransformer
from utils.comm import get_world_size, get_rank, is_main_process
from utils.deepspeed import get_deepspeed_config, fp32_to_fp16
import os.path as op

from utils.miscellaneous import concat_tsv_files, delete_tsv_files
from utils.tsv_file_ops import tsv_writer, reorder_tsv_keys
import torch.distributed as dist

class MVVC:
    def __init__(self, args):
        self.args = args

        self._build_model()
        self._build_dataloader()
        self._build_optimizer()


    def _build_dataloader(self):
        self.train_dataloader = make_data_loader(self.args, self.tokenizer_text, is_train=True)
        self.val_dataloader = make_data_loader(self.args, self.tokenizer_text, is_train=False)

    def _build_model(self):
        self.model = VideoTransformer(self.args)

        self.tokenizer_text = self.model.tokenizer
        # self.model.freeze_backbone(freeze=self.args.freeze_backbone)
        a = [p for p in self.model.named_parameters()]
        self.model.to(self.args.device)


    def _build_optimizer(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
        no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

        decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "extracter" in n]
        decay_bert_param_tp = [(n, p) for n, p in decay_param_tp if "extracter" not in n]

        no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "extracter" in n]
        no_decay_bert_param_tp = [(n, p) for n, p in no_decay_param_tp if  "extracter" not in n]

        optimizer_grouped_parameters = [
            # {'params': [p for n, p in decay_clip_param_tp],
            #  'weight_decay': self.args.weight_decay,
            #  'lr': self.args.learning_rate * self.args.backbone_coef_lr},
            #
            # {'params': [p for n, p in no_decay_clip_param_tp],
            #  'weight_decay': 0.0,
            #  'lr': self.args.learning_rate * self.args.backbone_coef_lr},

            {'params': [p for n, p in no_decay_bert_param_tp],
             'weight_decay': 0.0,
             'lr': self.args.learning_rate * self.args.backbone_coef_lr},
            {'params': [p for n, p in decay_bert_param_tp],
             'weight_decay': self.args.weight_decay,
             'lr': self.args.learning_rate * self.args.backbone_coef_lr},
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.args.learning_rate,
            eps=self.args.adam_epsilon)

        max_iter = len(self.train_dataloader)
        max_global_step =  max_iter// self.args.gradient_accumulation_steps
        if self.args.scheduler == "warmup_linear":
            self.scheduler = WarmupLinearLR(
                self.optimizer, max_global_step, warmup_ratio=self.args.warmup_ratio)
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=int(max_iter / 2.0), gamma=0.1)

        if self.args.mixed_precision_method == "deepspeed":
            config = get_deepspeed_config(self.args)
            self.model, self.optimizer, _, _ = deepspeed.initialize(
                config_params=config,
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.scheduler)

            if op.exists(op.join(self.args.output_dir, "model_dict.pth")):
                # load checkpoint
                _, self.client_state = self.model.load_checkpoint(op.join(self.args.output_dir, "model_dict.pth"))
                # step = self.client_sd['step']
                # dataloader_to_step(self.train_dataloader, step + 1)
            else:
                self.client_state = dict()

    def train(self):
        max_iter = len(self.train_dataloader)
        iters_per_epoch = max_iter // self.args.num_train_epochs
        result1, result2 = dict(), dict()


        for iteration, (img_keys, batch, meta_data) in enumerate(self.train_dataloader):
            iteration += 1

            # VTT, VVT = batch[0], batch[1]
            #
            # VTT = tuple(t.to(self.args.device) for t in VTT)
            # VVT = tuple(t.to(self.args.device) for t in VVT)

            VTT = tuple(t.to(self.args.device) for t in batch)
            self.model.train()

            inputs = {
                'input_ids': VTT[0], 'attention_mask': VTT[1],
                'token_type_ids': VTT[2], 'img_feats': VTT[3],
                'masked_pos': VTT[4], 'masked_ids': VTT[5],
                'img_feats_f': VTT[6],
            }
            if self.args.deepspeed_fp16:
                # deepspeed does not autocast inputs
                inputs = fp32_to_fp16(inputs)

            outputs = self.model(**inputs)
            loss, logits = outputs[:2]

            if self.args.learn_mask_enabled:
                loss_sparsity = outputs[-1]
                loss = loss + (loss_sparsity * self.args.loss_sparse_w)

            if self.args.mixed_precision_method == "deepspeed":
                self.model.backward(loss)

            backward_now = iteration % self.args.gradient_accumulation_steps == 0
            if backward_now:
                if self.args.mixed_precision_method == "deepspeed":
                    self.model.step()
                else:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

            if iteration % iters_per_epoch == 0 or iteration == max_iter:
                epoch = iteration // iters_per_epoch
                checkpoint_dir = op.join(self.args.output_dir, 'checkpoint-{}-{}'.format(epoch, iteration))
                result1_i, result2_i = self.evaluate(checkpoint_dir)
                result1[epoch],  result2[epoch]= result1_i, result2_i

                eval_result_observed = op.join(self.args.output_dir, "eval_result_1.json")
                with open(eval_result_observed, 'w') as f:
                    json.dump(result1, f)

                eval_result_unobserved = op.join(self.args.output_dir, "eval_result_2.json")
                with open(eval_result_unobserved, 'w') as f:
                    json.dump(result2, f)

                self.client_state['step'] = iteration
                # ckpt_id = loss.item()
                ckpt_id = 1
                self.model.save_checkpoint(op.join(self.args.output_dir, "model_dict.pth"), ckpt_id, client_state=self.client_state)

    def evaluate(self, checkpoint_dir):
        def gen_rows(predict_caption, cap_i = "observed"):
            cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id = \
                self.tokenizer_text.convert_tokens_to_ids([self.tokenizer_text.cls_token, self.tokenizer_text.sep_token,
                                                 self.tokenizer_text.pad_token, self.tokenizer_text.mask_token, '.'])

            self.model.eval()
            with torch.no_grad():
                for step, (img_keys, batch, meta_data) in enumerate(tqdm(self.val_dataloader)):

                    # VTT, VVT = batch[0], batch[1]
                    #
                    # VTT = tuple(t.to(self.args.device) for t in VTT)
                    # VVT = tuple(t.to(self.args.device) for t in VVT)

                    VTT = tuple(t.to(self.args.device) for t in batch)
                    inputs = {'is_decode': True,
                        'input_ids': VTT[0], 'attention_mask': VTT[1],
                        'token_type_ids': VTT[2], 'img_feats': VTT[3],
                        'masked_pos': VTT[4], 'img_feats_f': VTT[5],
                        'do_sample': False,
                        'bos_token_id': cls_token_id,
                        'pad_token_id': pad_token_id,
                        'eos_token_ids': [sep_token_id],
                        'mask_token_id': mask_token_id,
                        # for adding od labels
                        'add_od_labels': self.args.add_od_labels, 'od_labels_start_posid': self.args.max_seq_a_length,
                        # hyperparameters of beam search
                        'max_length': self.args.max_seq_a_length if not self.args.use_sep_cap else self.args.max_seq_length,
                        'use_sep_cap': self.args.use_sep_cap,
                        'num_beams': self.args.num_beams,
                        "temperature": self.args.temperature,
                        "top_k": self.args.top_k,
                        "top_p": self.args.top_p,
                        "repetition_penalty": self.args.repetition_penalty,
                        "length_penalty": self.args.length_penalty,
                        "num_return_sequences": self.args.num_return_sequences,
                        "num_keep_best": self.args.num_keep_best,
                    }

                    if self.args.deepspeed_fp16:
                        # deepspeed does not auto cast inputs.
                        inputs = fp32_to_fp16(inputs)

                    outputs = self.model(**inputs)

                    all_caps = outputs[0]  # batch_size * num_keep_best * max_len
                    all_confs = torch.exp(outputs[1])

                    if predict_caption != 2:
                        for img_key, caps, confs in zip(img_keys, all_caps, all_confs):
                            res = []
                            for cap, conf in zip(caps, confs):
                                cap = self.tokenizer_text.decode(cap.tolist(), skip_special_tokens=True)
                                res.append({'caption': cap, 'conf': conf.item()})
                            yield img_key, json.dumps(res)
                    else:
                        for img_key, caps, confs in zip(img_keys, all_caps, all_confs):
                            all_cap_a = []
                            all_cap_b = []
                            sep_place = self.args.max_seq_a_length
                            for cap, conf in zip(caps, confs):
                                cap_1 = self.tokenizer_text.decode(cap.tolist()[:sep_place], skip_special_tokens=True)
                                cap_2 = self.tokenizer_text.decode(cap.tolist()[sep_place:], skip_special_tokens=True)
                                all_cap_a.append({'caption': cap_1, 'conf': conf.item()})
                                all_cap_b.append({'caption': cap_2, 'conf': conf.item()})
                            if cap_i == "observed":
                                yield img_key, json.dumps(all_cap_a)
                            elif cap_i == "unobserved":
                                yield img_key, json.dumps(all_cap_b)

        if self.args.predict_caption == 0:
            predict_file = op.join(checkpoint_dir, "result.tsv")
            tsv_writer(gen_rows(0), predict_file)

            evaluate_file = op.splitext(predict_file)[0] + '.eval.json'
            caption_file = op.join("datasets", op.dirname(self.args.train_yaml), "val.caption_observed_coco_format.json")
            result = self.evaluator(caption_file, predict_file, evaluate_file)
            return result, None
        elif self.args.predict_caption == 1:
            predict_file = op.join(checkpoint_dir, "result.tsv")
            tsv_writer(gen_rows(1), predict_file)

            evaluate_file = op.splitext(predict_file)[0] + '.eval.json'
            caption_file = op.join("datasets", op.dirname(self.args.train_yaml), "val.caption_unobserved_coco_format.json")
            result = self.evaluator(caption_file, predict_file, evaluate_file)
            return result, None
        else:
            predict_file = op.join(checkpoint_dir, "observed_result.tsv")
            tsv_writer(gen_rows(2, cap_i = "observed"), predict_file)

            evaluate_file = op.splitext(predict_file)[0] + '.observed_eval.json'
            caption_file = op.join("datasets", op.dirname(self.args.train_yaml), "val.caption_observed_coco_format.json")
            result_observed = self.evaluator(caption_file, predict_file, evaluate_file)


            predict_file = op.join(checkpoint_dir, "unobserved_result.tsv")
            tsv_writer(gen_rows(2, cap_i = "unobserved"), predict_file)

            evaluate_file = op.splitext(predict_file)[0] + '.unobserved_eval.json'
            caption_file = op.join("datasets", op.dirname(self.args.train_yaml), "val.caption_unobserved_coco_format.json")
            result_unobserved = self.evaluator(caption_file, predict_file, evaluate_file)

            return result_observed, result_unobserved


    def test(self):
        pass

    def evaluator(self, caption_file, predict_file, output_file):
        predict_file_coco = op.splitext(predict_file)[0] + '_coco_format.json'
        convert_tsv_to_coco_format(predict_file, predict_file_coco)

        coco = COCO(caption_file)
        cocoRes = coco.loadRes(predict_file_coco)
        cocoEval = COCOEvalCap(coco, cocoRes, 'corpus')
        cocoEval.params['image_id'] = cocoRes.getImgIds()
        cocoEval.evaluate()
        result = cocoEval.eval

        with open(output_file, 'w') as fp:
            json.dump(result, fp, indent=4)

        return result

