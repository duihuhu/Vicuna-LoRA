# Usage: deepspeed train_lora.py --deepspeed <$PATH_TO_DEEPSPEED_CONFIG>

# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import logging
import pathlib
import typing
import os

from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
import transformers
from transformers import Trainer, BitsAndBytesConfig, deepspeed, LlamaForCausalLM
import torch

from fastchat.train.train import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
    make_supervised_data_module,
)
from fastchat.model.apply_lora import apply_lora


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: typing.List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def cli(model_path):
    from fastchat.serve.cli import chat_loop, SimpleChatIO, GptqConfig

    chat_loop(
        model_path,
        device="cuda",
        num_gpus=1,
        max_gpu_memory=None,
        load_8bit=None,
        cpu_offloading=None,
        conv_template=None,
        temperature=0.7,
        repetition_penalty=1.0,
        max_new_tokens=512,
        chatio=SimpleChatIO(),
        gptq_config=GptqConfig(
            ckpt=None,
            wbits=16,
            groupsize=-1,
            act_order=None,
        ),
        revision="main",
        judge_sent_end=False,
        debug=None,
    )


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    model = transformers.OPTForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        # load_in_8bit=True,
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_int8_training(model)
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    if training_args.deepspeed is not None and training_args.local_rank == 0:
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # for d in data_module['train_dataset']:
    #     print(d)
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    model.config.use_cache = False

    lora_path = os.path.join(training_args.output_dir, "lora")
    if list(pathlib.Path(lora_path).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    # check if zero3 mode enabled
    if trainer.hf_deepspeed_config_orig.is_zero3():
        # use deepspeed engine internal function to gather state dict
        # state_dict_zero3 contains whole parameters of base and lora adapters
        # we will not extract lora parameters since peft save_pretrained will do that
        # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/peft_model.py#L125
        # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/utils/save_and_load.py#L19
        state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
        if training_args.local_rank == 0:
            state_dict = state_dict_zero3
    else:
        # in other mode we use original code from fastchat team, to make sure our change is minimum
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), lora_args.lora_bias
        )

    if training_args.local_rank == 0:
        model.save_pretrained(lora_path, state_dict=state_dict)

        # apply lora
        model_path = os.path.join(training_args.output_dir, "model")
        apply_lora(model_args.model_name_or_path, model_path, lora_path)

        # generate
        print("==== start chatting ===")
        cli(model_path)


if __name__ == "__main__":
    train()
