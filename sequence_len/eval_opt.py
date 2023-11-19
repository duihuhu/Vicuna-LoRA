import torch
import tqdm
import transformers
from fastchat.serve.inference import load_model
from peft import PeftModel
import utils

if __name__ == "__main__":
    # data
    data = utils.EvalDataset("../alpaca-val-10k_part.json")
    data_len = utils.jload("../alpaca-val-10k-length_part.json")
    temp = [0.0, 0.3, 0.5, 0.7]

    for i in range(len(data)):
        assert data[i]["id"] == data_len[i]["id"], f"{data[i]['id']} != {data_len[i]['id']}"
        data_len[i]["L_max"] = max([data_len[i][f"L_t{t}"] for t in temp])

    # model
    model, tokenizer = load_model(
        "/workspace/stanford_alpaca/model",
        "cuda",
        1,
        load_8bit=None,
        debug=False,
        revision="main",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "/workspace/stanford_alpaca/model",
        model_max_length=256,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "left"

    # LORA
    load_lora = "/workspace/Sequence-Scheduling/ckpts/lora"
    length_predictor = PeftModel.from_pretrained(
        model, load_lora, torch_dtype=torch.float16
    )
    print(f"Loaded LORA length predictor {load_lora}")

    # Prompt
    PROMPT_QUERY = "\nDon't output the response for the above instruction. Instead, you need to predict the number of tokens in your response. Output one number only."

    # eval
    N = 2
    batch_size = 2
    cnt = 0
    all_outputs = []
    gt_order_outputs = []
    diff = 0
    acc_50t = 0
    acc_100t = 0

    for i in tqdm.tqdm(range(0, N, batch_size)):
        batch = data[i : i + batch_size]
        inputs = batch["input"]
        labels = data_len[i : i + batch_size]
        cnt += len(inputs)

        # Inference
        inputs_ = [p[:-11] + PROMPT_QUERY + p[-11:] for p in inputs]
        inputs_ = tokenizer(inputs_, return_tensors="pt", padding=True)

        l_prompt = inputs_["input_ids"].shape[1]
        input_ids = inputs_["input_ids"].cuda()
        attn_mask = inputs_["attention_mask"].cuda()
        output_ids = length_predictor.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=512,
            do_sample=False,
        )

        outputs = [
            tokenizer.decode(x[l_prompt:], skip_special_tokens=True) for x in output_ids
        ]
        print("outputs: ",outputs)
        
        outputs = [int(x.strip()) for x in outputs]
        labels_max = [x["L_max"] for x in labels]
        

        # collect results
        all_outputs.extend(outputs)
        gt_order_outputs.extend(labels_max)
        d_max = [abs(x - y) for x, y in zip(outputs, labels_max)]
        diff += sum(d_max)
        acc_50t += sum([1 if x <= 50 else 0 for x in d_max])
        acc_100t += sum([1 if x <= 100 else 0 for x in d_max])

    print(f"# Samples: {cnt}")
    print(f"Error: {diff / cnt}")
    print(f"Acc-50: {acc_50t / cnt}")
    print(f"Acc-100: {acc_100t / cnt}")
