from logging import raiseExceptions
import torch
import random
import re, html

FLAIR_TAG = {
    "noun": ["NN", "NNP", "NNPS", "NNS", "PRP", "PRP$", "WP", "WP$"],
    "verb": ["VB", "VBD", "VBG", "VBP", "VBZ"],
    "adjective": ["JJ", "JJR", "JJS"],
    "adverb": ["RB","RBR", "RBS", "WRB"],
    "number": ["CD"]}


class CaptionTensorizer(object):
    def __init__(self, tokenizer, max_img_seq_length=50, max_seq_length=70, 
            max_seq_a_length=40, mask_prob=0.15, max_masked_tokens=3,
            attn_mask_type='seq2seq', is_train=True, mask_b=False, attn_mask_compose=None,
            text_mask_type='random', tag_to_mask=None,
            mask_tag_prob=0.8, random_mask_prob=0.5, use_sep_cap = False):
        """Constructor.
        Args:
            tokenizer: tokenizer for text processing.
            max_img_seq_length: max image sequence length.
            max_seq_length: max text sequence length.
            max_seq_a_length: max caption sequence length.
            is_train: train or test mode.
            mask_prob: probability to mask a input token.
            max_masked_tokens: maximum number of tokens to be masked in one sentence.
            attn_mask_type: attention mask type, support seq2seq/bidirectional/cap_s2s/cap_bidir.
            mask_b: whether to mask text_b or not during training.
        """
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_img_seq_len = max_img_seq_length
        self.max_seq_len = max_seq_length
        self.max_seq_a_len = max_seq_a_length
        self.mask_prob = mask_prob
        self.max_masked_tokens = max_masked_tokens
        self.attn_mask_type = attn_mask_type
        self.attn_mask_compose = attn_mask_compose
        self.text_mask_type = text_mask_type
        self.mask_b = use_sep_cap
        self.tag_to_mask = None
        self.mask_tag_prob = 0
        self.random_mask_prob = 1
        self.use_sep_cap = use_sep_cap
        if is_train:
            assert attn_mask_type in ('seq2seq', 'bidirectional', 'cap_s2s', 'cap_bidir', 'learn_vid_att', 'learn_without_crossattn', 'learn_with_swap_crossattn')
            assert text_mask_type in ('random', 'bert_attn', 'pos_tag', 'attn_on_the_fly')
            if self.text_mask_type == 'pos_tag':
                self.tag_to_mask = tag_to_mask
                self.included_tags = set()
                for type in self.tag_to_mask:
                    self.included_tags.update(set(FLAIR_TAG[type]))
                self.mask_tag_prob = mask_tag_prob
            if self.text_mask_type != "random":
                self.random_mask_prob = random_mask_prob
        else:
            assert attn_mask_type in ('seq2seq', 'learn_vid_att', 'learn_without_crossattn', 'learn_with_swap_crossattn')
        
        self._triangle_mask = torch.tril(torch.ones((self.max_seq_len, 
            self.max_seq_len), dtype=torch.long))
    
    def get_pos_tag_mask_idx(self, seq_a_len, text_meta):
        
        ''' The rest   
        ADD	Email
        AFX	Affix
        CC	Coordinating conjunction
        DT	Determiner
        EX	Existential there
        FW	Foreign word
        HYPH	Hyphen
        IN	Preposition or subordinating conjunction
        LS	List item marker
        MD	Modal
        NFP	Superfluous punctuation
        PDT	Predeterminer
        POS	Possessive ending
        RP	Particle
        SYM	Symbol
        TO	to
        UH	Interjection
        WDT	Wh-determiner
        XX
        '''
        # process loaded pos_tags
        pos_tags =  text_meta["bert_pos_tag"] 
        if len(pos_tags) > seq_a_len - 2:
            pos_tags = pos_tags[:seq_a_len-2]
        pos_tags = [None] + pos_tags + [None]
        padding_len = seq_a_len - len(pos_tags)
        pos_tags += [None] * padding_len
        allow_masked_ids = set()
        for bert_idx, tag in enumerate(pos_tags):
            if tag is None:
                continue
            if bert_idx >= seq_a_len:
                break
            if tag not in self.included_tags:
                continue
            allow_masked_ids.add(bert_idx)
        return pos_tags, allow_masked_ids
    
    def get_bert_attn_mask_idx(self, seq_a_len, text_meta, num_masked):
        # process loaded bert attention weights (assuming max_len = 50)
        attn_weights =  text_meta["bert_attn"] 
        if len(attn_weights) > seq_a_len:
            attn_weights = attn_weights[:seq_a_len]
        elif len(attn_weights) < seq_a_len:
            # pad with zeros
            padding_len = seq_a_len - len(attn_weights)
            attn_weights = [0.0] * padding_len
        mask_idx = torch.multinomial(torch.tensor(attn_weights), num_masked).tolist()
        return mask_idx

    def get_attn_masks(self, seq_a_len, seq_b_len):
        # image features
        img_len = self.max_img_seq_len

        max_len = self.max_seq_len + self.max_img_seq_len * 2
        # C: caption, L: label, R: image region
        cc_start, cc_end = 0, seq_a_len
        fc_start, fc_end = self.max_seq_a_len, self.max_seq_a_len + seq_b_len
        cv_start, cv_end = self.max_seq_len, self.max_seq_len + img_len
        fv_start, fv_end = self.max_seq_len + img_len, self.max_seq_len + 2 * img_len

        attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
        # triangle mask for caption_a to caption_a
        attention_mask[cc_start: cc_end, cc_start: cc_end].copy_(
            self._triangle_mask[0: seq_a_len, 0: seq_a_len]
        )

        # triangle mask for caption_b to caption_b
        attention_mask[fc_start: fc_end, fc_start: fc_end].copy_(
            self._triangle_mask[0: seq_b_len, 0: seq_b_len]
        )

        # # full attention for C_b-C_a
        # attention_mask[fc_start: fc_end, cc_start: cc_end] = 1
        #
        # # full attention for C_a-R, C_b-R
        # attention_mask[cc_start: cc_end, cv_start: cv_end] = 1
        # attention_mask[fc_start: fc_end, cv_start: cv_end] = 1

        # full attention for video tokens:
        attention_mask[cv_start: cv_end, cv_start: cv_end] = 1

        if self.attn_mask_compose.cc_fc:
            attention_mask[cc_start: cc_end, fc_start: fc_end] = 1
        if self.attn_mask_compose.cc_fv:
            attention_mask[cc_start: cc_end, fv_start: fv_end] = 1
        if self.attn_mask_compose.cc_cv:
            attention_mask[cc_start: cc_end, cv_start: cv_end] = 1
        if self.attn_mask_compose.fc_cc:
            attention_mask[fc_start: fc_end, cc_start: cc_end] = 1
        if self.attn_mask_compose.fc_fv:
            attention_mask[fc_start: fc_end, fv_start: fv_end] = 1
        if self.attn_mask_compose.fc_cv:
            attention_mask[fc_start: fc_end, cv_start: cv_end] = 1
        if self.attn_mask_compose.fv_cc:
            attention_mask[fv_start: fv_end, cc_start: cc_end] = 1
        if self.attn_mask_compose.fv_fc:
            attention_mask[fv_start: fv_end, fc_start: fc_end] = 1
        if self.attn_mask_compose.fv_cv:
            attention_mask[fv_start: fv_end, cv_start: cv_end] = 1

        return attention_mask


    def get_text_mask_idx(self, seq_a_len, seq_b_len, text_meta=None):
        # randomly mask words for prediction, ignore [CLS], [PAD]
        # it is important to mask [SEP] for image captioning as it means [EOS].

        # 1. get the number of masked tokens
        if self.mask_b:
            # can mask both text_a and text_b
            num_masked = min(max(round(self.mask_prob * (seq_a_len+seq_b_len)), 1), self.max_masked_tokens)
        else:
            # only mask text_a
            num_masked = min(max(round(self.mask_prob * seq_a_len), 1), self.max_masked_tokens)
        num_masked = int(num_masked)

        # 2. get the masking candidates
        if self.mask_b:
            # text b always random masking
            text_b_candidate = list(range(self.max_seq_a_len+1, self.max_seq_a_len+seq_b_len))
        else:
            text_b_candidate = []
        if self.text_mask_type == 'random':
            # random
            candidate_masked_idx = list(range(1, seq_a_len))
            candidate_masked_idx += text_b_candidate
            random.shuffle(candidate_masked_idx)
            masked_idx = candidate_masked_idx[:num_masked]
        else:
            raiseExceptions("text_mask_type is not random")
        masked_idx = sorted(masked_idx)
        return masked_idx
    
    def mask_text_inputs(self, tokens, seq_a_len, seq_b_len, text_meta=None):
        if self.is_train:
            if self.text_mask_type == "attn_on_the_fly" and random.random() > self.random_mask_prob and len(tokens)> 2:
                # self.text_mask_type == "attn_on_the_fly"
                masked_pos = torch.zeros(self.max_seq_len, dtype=torch.int)
                masked_pos[1: seq_a_len] += 1
                masked_pos[0] = self.tokenizer.convert_tokens_to_ids([self.tokenizer.mask_token])[0]
                mlm_targets = [-1] * self.max_masked_tokens
            else:
                masked_idx = self.get_text_mask_idx(seq_a_len, seq_b_len, text_meta)
                try:
                    masked_token = [tokens[i] for i in masked_idx]
                except Exception as e:
                    overflow_idx = []
                    for i in masked_idx:
                        if i >= len(tokens) or i < 0:
                            overflow_idx.append(i)
                    raise ValueError(f"Error {e}\nOverflow: {overflow_idx} in tokens {tokens}")
                for pos in masked_idx:
                    if random.random() <= 0.8:
                        # 80% chance to be a ['MASK'] token
                        tokens[pos] = self.tokenizer.mask_token
                    elif random.random() <= 0.5:
                        # 10% chance to be a random word ((1-0.8)*0.5)
                        tokens[pos] = self.tokenizer.get_random_token()
                    else:
                        # 10% chance to remain the same (1-0.8-0.1)
                        pass

                masked_pos = torch.zeros(self.max_seq_len, dtype=torch.int)
                masked_pos[masked_idx] = 1
                
                # get the actual number of masked tokens
                num_masked = len(masked_token)
                mlm_targets = self.tokenizer.convert_tokens_to_ids(masked_token)
                if num_masked < self.max_masked_tokens:
                    mlm_targets = mlm_targets + ([-1] * (self.max_masked_tokens - num_masked))
                assert len(mlm_targets) == self.max_masked_tokens, f"mismatch in len(masked_ids) {len(mlm_targets)} vs. max_masked_tokens {self.max_masked_tokens}"
        elif not self.is_train:
            masked_pos = torch.ones(self.max_seq_len, dtype=torch.int)
            mlm_targets = None
        
        return tokens, masked_pos, mlm_targets
    
    def prepro_raw_txt(self, text):
        # in case there are html special characters
        text = html.unescape(text)
        # FIXME: quick hack for text with emoji, may adopt twitter tokenizer later
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        return text
    
    def tokenize_text_inputs(
            self, text_a, text_b=None, cls_token_segment_id=0,
            pad_token_segment_id=0, sequence_a_segment_id=0,
            sequence_b_segment_id=1, text_meta=None):
        text_a = self.prepro_raw_txt(text_a)
        if self.is_train:
            tokens_a = self.tokenizer.tokenize(text_a)
        else:
            # fake tokens to generate masks
            tokens_a = [self.tokenizer.mask_token] * (self.max_seq_a_len - 2)
        if len(tokens_a) > self.max_seq_a_len - 2:
            tokens_a = tokens_a[:(self.max_seq_a_len - 2)]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
        seq_a_len = len(tokens)

        padding_a_len = self.max_seq_a_len - seq_a_len
        tokens += [self.tokenizer.pad_token] * padding_a_len
        segment_ids += ([pad_token_segment_id] * padding_a_len)


        seq_b_len = 0
        if text_b is not None:
            # pad text_a to keep it in fixed length for better inference.

            text_b = self.prepro_raw_txt(text_b)
            if self.is_train:
                tokens_b = self.tokenizer.tokenize(text_b)
            else:
                tokens_b = [self.tokenizer.mask_token] * (self.max_seq_a_len - 2)
            if len(tokens_b) > self.max_seq_len - len(tokens) - 2:
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 2)]
            tokens_b = [self.tokenizer.cls_token] + tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b))
            seq_b_len = len(tokens_b)
            tokens += tokens_b

            seq_len = len(tokens)
            padding_b_len = self.max_seq_len - seq_len
            tokens += [self.tokenizer.pad_token] * padding_b_len
            segment_ids += ([sequence_b_segment_id] * padding_b_len)

        
        return tokens, segment_ids, seq_a_len, seq_b_len

    def tensorize_example_VTT(self, text_a, img_feat, text_b=None,
            cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1, text_meta=None):
        # tokenize the texts
        tokens, segment_ids, seq_a_len, seq_b_len = self.tokenize_text_inputs(
            text_a, text_b, cls_token_segment_id, pad_token_segment_id,
            sequence_a_segment_id, sequence_b_segment_id, text_meta)
        
        # masking the tokens
        tokens_after_masking, masked_pos, mlm_targets = self.mask_text_inputs(
            tokens, seq_a_len, seq_b_len, text_meta)

        # pad on the right for image captioning
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens_after_masking)

        attention_mask = self.get_attn_masks(seq_a_len, seq_b_len)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        if self.is_train:
            mlm_targets = torch.tensor(mlm_targets, dtype=torch.long)
            return (input_ids, attention_mask, segment_ids, img_feat, masked_pos, mlm_targets)
        return input_ids, attention_mask, segment_ids, img_feat, masked_pos

    def tensorize_example_VVT(self, text_a, img_feat, text_b=None, img_feat_f=None,
                              cls_token_segment_id=0, pad_token_segment_id=0,
                              sequence_a_segment_id=0, sequence_b_segment_id=1, text_meta=None):
        # tokenize the texts
        tokens, segment_ids, seq_a_len, seq_b_len = self.tokenize_text_inputs(
            text_a, None, cls_token_segment_id, pad_token_segment_id,
            sequence_a_segment_id, sequence_b_segment_id, text_meta)

        # masking the tokens
        tokens_after_masking, masked_pos, mlm_targets = self.mask_text_inputs(
            tokens, seq_a_len, seq_b_len, text_meta)

        # pad on the right for image captioning
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens_after_masking)

        attention_mask = self.get_attn_masks(seq_a_len, seq_b_len)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        if self.is_train:
            mlm_targets = torch.tensor(mlm_targets, dtype=torch.long)
            return (input_ids, attention_mask, segment_ids, img_feat, masked_pos, mlm_targets, img_feat_f)
        return input_ids, attention_mask, segment_ids, img_feat, masked_pos


    def tensorize_example_VVTT(self, text_a, img_feat, text_b=None, img_feat_f=None,
                              cls_token_segment_id=0, pad_token_segment_id=0,
                              sequence_a_segment_id=0, sequence_b_segment_id=1, text_meta=None):

        # img_feat = torch.cat((img_feat, img_feat_f), 0)

        # tokenize the texts
        tokens, segment_ids, seq_a_len, seq_b_len = self.tokenize_text_inputs(
            text_a, text_b, cls_token_segment_id, pad_token_segment_id,
            sequence_a_segment_id, sequence_b_segment_id, text_meta)

        # masking the tokens
        tokens_after_masking, masked_pos, mlm_targets = self.mask_text_inputs(
            tokens, seq_a_len, seq_b_len, text_meta)

        # pad on the right for image captioning
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens_after_masking)

        attention_mask = self.get_attn_masks(seq_a_len, seq_b_len)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        if self.is_train:
            mlm_targets = torch.tensor(mlm_targets, dtype=torch.long)
            return (input_ids, attention_mask, segment_ids, img_feat, masked_pos, mlm_targets, img_feat_f)
        return input_ids, attention_mask, segment_ids, img_feat, masked_pos, img_feat_f

def build_tensorizer(args, tokenizer, is_train):
    if hasattr(args, 'mask_od_labels'):
        mask_b = args.mask_od_labels
    else:
        mask_b = False
    if is_train:
        if  args.text_mask_type == "pos_tag":
            tag_to_mask = set(args.tag_to_mask)
        else:
            tagger = None
            tag_to_mask = None
        return CaptionTensorizer(
            tokenizer,

            # max_img_seq_length=args.max_img_seq_length,
            # max_img_seq_length=args.max_img_seq_length + args.max_num_frames,
            # max_img_seq_length=args.max_num_frames * 2,
            max_img_seq_length=args.max_num_frames,
            # max_img_seq_length=args.max_img_seq_length * 2 + args.max_num_frames * 2,
            max_seq_length=args.max_seq_length,
            max_seq_a_length=args.max_seq_a_length,
            mask_prob=args.mask_prob,
            max_masked_tokens=args.max_masked_tokens,
            attn_mask_type=args.attn_mask_type,
            attn_mask_compose=args.attn_mask_compose,
            is_train=True,
            mask_b=mask_b,
            text_mask_type=args.text_mask_type,
            mask_tag_prob=args.mask_tag_prob,
            tag_to_mask=tag_to_mask,
            random_mask_prob=args.random_mask_prob,
            # tagger=tagger,
            use_sep_cap = args.use_sep_cap,
        )
    return CaptionTensorizer(
            tokenizer,
            # max_img_seq_length=args.max_img_seq_length * 2 + args.max_num_frames * 2,
            # max_img_seq_length=args.max_img_seq_length + args.max_num_frames,
            max_img_seq_length=args.max_num_frames,
            # max_img_seq_length=args.max_num_frames * 2,
            # max_img_seq_length=args.max_img_seq_length,
            max_seq_length=args.max_seq_length,
            max_seq_a_length=args.max_seq_a_length,
            is_train=False,
            attn_mask_type=args.attn_mask_type,
            attn_mask_compose=args.attn_mask_compose,
            use_sep_cap = args.use_sep_cap,
    )
