import torch
from torch.nn import CrossEntropyLoss
from transformers import CLIPVisionModel, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from dataset.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_TOKEN, \
    DEFAULT_PAD_TOKEN, DEFAULT_IM_PATCH_TOKEN, DEFAULT_BBOX_X_TOKEN, DEFAULT_BBOX_Y_TOKEN, \
    DEFAULT_BBOX_W_TOKEN, DEFAULT_BBOX_H_TOKEN, DEFAULT_EMO_TOKEN

from model.topk import PatchNet
from model.atp_lan import ATPSelectorModel, ATPConfig
class ViTLlamaModel(LlamaModel):
    def __init__(self, config):
        super().__init__(config)


        if hasattr(self.config, 'vit_path'):
            self.vit = CLIPVisionModel.from_pretrained(self.config.vit_path)

        if hasattr(self.config, 'vit_hidden_size'):
            self.llm_proj = torch.nn.Linear(config.vit_hidden_size, self.config.hidden_size)

        if hasattr(self.config, "use_delta_transformer") and config.use_delta_transformer:
            print('Using temporal transformer delta adding')
            self.transformer_adding_layer = torch.nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=8, batch_first=True)
            self.transformer_delta_encoder = torch.nn.TransformerEncoder(self.transformer_adding_layer, num_layers=1)
            self.config.patch_pooling_method = "temporal_transformer"
            self.position_matrix = torch.nn.Parameter(torch.zeros(500, config.hidden_size))
            self.position_matrix.requires_grad = False

        if hasattr(self.config, "use_atp") and config.use_atp:
            print('Using atp adding')
            atp_config = ATPConfig(use_ste=False, use_pe=True, soft_inference= True)
            print("creating ATPSelectorModel from config...")
            self.atp_model = ATPSelectorModel(atp_config)

        if hasattr(self.config, "use_tok_se") and config.use_tok_se:
            print('Using Patchnet')

            self.patch_net_tem = PatchNet(score='tpool', k=int(0.375*6), in_channels = 4096)
            self.patch_net_spa = PatchNet(score='spatch', k=int(0.7657*64), in_channels = 4096)
            print('Loaded Patchnet')


        if hasattr(self.config, "use_tube") and config.use_tube:
            print('Using Patchnet')
            self.select = 4
            self.patch_net_tube = PatchNet(score='tube', k=self.select, in_channels = 4096)

            print('Loaded Patchnet')



    
    def init_vision(self, vit_path, tokenizer_len, use_im_start_end=True, use_im_patches=True, use_bbox_tokens=False,
                    use_delta_transformer=False, use_atp=False, use_tok_se=False, use_lan=False, use_tube=False):
        if not hasattr(self, 'vit'):
            print('Loading ViT model')
            self.vit = CLIPVisionModel.from_pretrained(vit_path)
            print('Successfully loaded ViT model')
            self.config.vit_path = vit_path

        self.resize_token_embeddings(tokenizer_len)

        if not hasattr(self, 'use_delta_transformer') and use_delta_transformer:
            print('Creating temporal transformer delta encoder')
            self.transformer_adding_layer = torch.nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=8, batch_first=True)
            self.transformer_delta_encoder = torch.nn.TransformerEncoder(self.transformer_adding_layer, num_layers=1)
            self.config.patch_pooling_method = "temporal_transformer"
            self.position_matrix = torch.nn.Parameter(self.getPositionEncoding(seq_len=500, d=self.config.hidden_size))
            self.position_matrix.requires_grad = False

        if use_tube:
            print('Using Tube Patchnet')
            self.select = 4
            self.patch_net_tube = PatchNet(score='tube', k=self.select, in_channels = 4096)
            print('Loaded Tube Patchnet')

        if not hasattr(self, 'llm_proj'):
            print('Creating projection layer')
            self.llm_proj = torch.nn.Linear(self.vit.config.hidden_size, self.config.hidden_size)
            self.config.vit_hidden_size = self.vit.config.hidden_size

        
        
        self.config.use_im_patches = use_im_patches
        self.config.use_im_start_end = use_im_start_end
        self.config.use_bbox_tokens = use_bbox_tokens
        self.config.use_delta_transformer = use_delta_transformer
        self.config.use_atp = use_atp
        self.config.use_tok_se = use_tok_se
        self.config.use_tube = use_tube

    def getPositionEncoding(self, seq_len=2048, d=5120, n=10000):
        P = torch.zeros((seq_len, d))
        for k in range(seq_len):
            for i in torch.arange(int(d/2)):
                denominator = torch.pow(n, 2*i/d)
                P[k, 2*i] = torch.sin(k/denominator)
                P[k, 2*i+1] = torch.cos(k/denominator)
        return P

    def temporal_transformer_delta_adding(self, patch_feature):# 8, 256, 5120
        patch_feature = patch_feature.permute(1, 0, 2) # 256,8,5120
        sequence_length = patch_feature.shape[1]
        patch_number = patch_feature.shape[0]
        position_embedding = self.position_matrix[:sequence_length,:].unsqueeze(0).type_as(patch_feature)# 1,8,5120
        position_embedding = position_embedding.repeat(patch_number, 1, 1).to(patch_feature.device) # 256,8,5120
        patch_feature_pos = patch_feature + position_embedding
        patch_feature_delta = self.transformer_delta_encoder(patch_feature_pos)[:,-1,:]
        patch_feature_mean = torch.mean(patch_feature, dim=1) # 256 , 4096
        patch_feature = patch_feature_delta + patch_feature_mean
        return patch_feature
        

    def forward(self, images=None, input_ids=None, inputs_embeds=None, use_img=True, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)


        # outputs = tokenizer.batch_decode(output_ids skip_special_tokens=not args.use_bbox_tokens)
        assert images is not None
        if type(use_img) is bool:
            use_img = [use_img]
        if use_img[0]:
            if images.dim() == 5:
                # Multiple images (temporal)
                features = []
                for image in images:

                    features.append(self.vit(image, output_hidden_states=True).hidden_states[-2])
                    # Using the second to last hidden layer
                features = torch.stack(features, dim=0)
                num_frames = features.shape[1]

                if hasattr(self.config, "use_delta_transformer") and self.config.use_delta_transformer:
                    self.config.patch_pooling_method = 'temporal_transformer'
                elif hasattr(self.config, "use_tube") and self.config.use_tube:
                    self.config.patch_pooling_method = 'tube'
                else:
                    self.config.patch_pooling_method = 'mean'
            else:
                # Single image
                features = self.vit(images, output_hidden_states=True).hidden_states[-2]
                # Using the second to last hidden layer
                num_frames = 1

            # Project image features
                
            features_proj = self.llm_proj(features)

            if self.config.use_im_patches:
                if hasattr(self.config, "use_tok_se") and self.config.use_tok_se:
                    num_patches = 98
                elif hasattr(self.config, "use_tube") and self.config.use_tube:
                    num_patches = self.select * 32  
                    # num_patches = self.select * 18
                else:
                    num_patches = features_proj[...,1:,:].shape[-2]
            else:
                # Only using [CLS] feature
                features_proj = features_proj[...,0,:].unsqueeze(-2)
                num_patches = 0

            new_inputs_embeds = []

            dummy_image_features = torch.zeros(257, 1024, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_image_features = self.llm_proj(dummy_image_features)

            # Iterating over the batch
            for i, (cur_input_ids, cur_inputs_embeds) in enumerate(zip(input_ids, inputs_embeds)):

                if (cur_input_ids == self.config.image_token_id).sum() == 0:
                    # Used during inference
                    cur_inputs_embeds = cur_inputs_embeds + (0. * dummy_image_features).sum()
                    new_inputs_embeds.append(cur_inputs_embeds)
                    continue
                # Find the index of the image token
                image_tokens_idx = torch.where(cur_input_ids == self.config.image_token_id)[0][0]
                if self.config.use_im_start_end and \
                    cur_input_ids[image_tokens_idx + num_patches + num_frames] != self.config.im_end_token_id:
                    raise ValueError('Image tokens seem to be cut off')
                
                cur_new_inputs_embeds = cur_inputs_embeds.clone()

                cur_image_features = features_proj[i]
                if cur_image_features.dim() == 3:
                    # Multiple images (temporal)
                    # Get mean patch features and frame features
                    if self.config.patch_pooling_method == 'mean':
                        mean_image_features = torch.mean(cur_image_features[:,1:,:], dim=0)
                    elif self.config.patch_pooling_method == 'temporal_transformer':
                        mean_image_features = self.temporal_transformer_delta_adding(cur_image_features[:,1:,:])



                    elif self.config.patch_pooling_method == 'tube':
                        image_features = cur_image_features[:,1:,:].unsqueeze(0)
                        T = image_features.size(1)
                        select_mask= self.patch_net_tube(image_features, 'tube', 256, T, 0.1)
                        mean_image_features = select_mask.view(-1, 4096)

                    frame_image_features = cur_image_features[:,0,:]

                    # Combine both features
                    cur_image_features = torch.cat((frame_image_features, mean_image_features), dim=0)

                if self.config.use_im_start_end and \
                    ((cur_input_ids == self.config.im_start_token_id).sum() != (cur_input_ids == self.config.im_end_token_id).sum()):
                    raise ValueError('Mismatched number of image start and end tokens')


                # Place image feainit_vision
                new_inputs_embeds.append(torch.cat((cur_new_inputs_embeds[:image_tokens_idx], cur_image_features,
                                cur_new_inputs_embeds[image_tokens_idx + num_patches + num_frames:]), dim=0))
                
            new_inputs_embeds = torch.stack(new_inputs_embeds, dim=0)
        else:
            new_inputs_embeds = inputs_embeds
        if 'gt' in kwargs:
            del kwargs['gt']
        if 'videoid' in kwargs:
            del kwargs['videoid']
        return super().forward(input_ids=None, inputs_embeds=new_inputs_embeds, **kwargs)#, ini_score, score
    

class ViTLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        self.model = ViTLlamaModel(config)

        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()
    
    def init_vision(self, vit_path, tokenizer, use_im_start_end=True, use_im_patches=True, use_bbox_tokens=False, use_delta_transformer=False, use_img=True, use_emo_tokens=False, use_atp = False, use_lan = False, use_tok_se=False, use_tube=False):

        self.init_vision_tokenizer(tokenizer, use_im_start_end=use_im_start_end, use_im_patches=use_im_patches, use_bbox_tokens=use_bbox_tokens, use_emo_tokens=use_emo_tokens)

        tokenizer_len = len(tokenizer)
        if use_img:
            self.model.init_vision(vit_path, tokenizer_len, use_im_start_end=use_im_start_end, use_im_patches=use_im_patches, use_tube=use_tube,
                               use_bbox_tokens=use_bbox_tokens, use_delta_transformer=use_delta_transformer, use_atp=use_atp, use_lan = use_lan, use_tok_se=use_tok_se)


    def init_vision_tokenizer(self, tokenizer, use_im_start_end=True, use_im_patches=True, use_bbox_tokens=False, use_emo_tokens=False):
        model_config = self.get_model().config
        num_new_tokens = 0
        if tokenizer.pad_token is None:
            num_new_tokens += tokenizer.add_special_tokens({'pad_token': DEFAULT_PAD_TOKEN})
            tokenizer.pad_token = DEFAULT_PAD_TOKEN
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_PAD_TOKEN)
            print('Added pad token to tokenizer')

        num_new_tokens += tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN], special_tokens=True)
        tokenizer.image_token = DEFAULT_IMAGE_TOKEN
        tokenizer.image_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        model_config.image_token_id = tokenizer.image_token_id

        if use_im_start_end:
            num_new_tokens += tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            tokenizer.im_start_token = DEFAULT_IM_START_TOKEN
            tokenizer.im_end_token = DEFAULT_IM_END_TOKEN
            model_config.im_start_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)
            model_config.im_end_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
            print('Added image start and end tokens to tokenizer')
        
        if use_im_patches:
            num_new_tokens += tokenizer.add_tokens([DEFAULT_IM_PATCH_TOKEN], special_tokens=True)
            tokenizer.im_patch_token = DEFAULT_IM_PATCH_TOKEN
            model_config.im_patch_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IM_PATCH_TOKEN)
            print('Added image patch token to tokenizer')

        self.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

    def get_model(self):
        return self.model
    
    def forward(self, images=None,  input_ids=None, labels=None, use_img = True, **kwargs):
        # outputs, ini_score, score = self.model(images=images,
        outputs = self.model(images=images,
                             input_ids=input_ids,
                             use_img=use_img,
                             **kwargs)

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )#, ini_score, score

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "use_img": kwargs.get("use_img", None),
            }
        )
        return model_inputs