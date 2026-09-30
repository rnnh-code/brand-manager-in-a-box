"""Write notebooks/train_brand_manager.ipynb: the Colab notebook Rai runs top to bottom on an A100."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRIVE_DIR = "/content/drive/MyDrive/BrandManagerAI"

cells = [
    ("md", "# Brand Manager in a Box: fine-tune Gemma 4 12B\n\n"
           "**Before you start:** Runtime > Change runtime type > **A100 GPU**, then Runtime > **Run all**.\n\n"
           "Keep this tab open until the last cell finishes (about 2 to 3 hours). Progress is saved to your "
           "Google Drive folder `BrandManagerAI`, so a disconnect only loses the last few minutes."),
    ("code", "# True = 10-minute test run on the small 'smoke' set. False = the real training run.\n"
             "SMOKE_TEST = False\n"
             "from google.colab import drive\n"
             "drive.mount('/content/drive')\n"
             f"DIR = '{DRIVE_DIR}' + ('/smoke' if SMOKE_TEST else '')\n"
             "import os; print(sorted(os.listdir(DIR)))"),
    # Gemma 4 12B is model type gemma4_unified, which needs transformers 5.10; Unsloth pins <=5.5.
    # Unsloth Studio uses 5.10.2 for this model (unslothai/unsloth PR 6054), so we do the same.
    ("code", "%%capture\n!pip install -U unsloth\n!pip install 'transformers==5.10.2'"),
    ("md", "## 1. Load the starter brain\n16-bit LoRA: we train a small add-on (about 1% of the model) instead of the whole thing."),
    ("code", "from unsloth import FastModel\n"
             "model, tokenizer = FastModel.from_pretrained(\n"
             "    model_name = 'unsloth/gemma-4-12b-it',\n"
             "    max_seq_length = 2048,\n"
             "    load_in_4bit = False,\n"
             "    full_finetuning = False,\n"
             "    use_gradient_checkpointing = 'unsloth',\n"
             ")\n"
             "model = FastModel.get_peft_model(\n"
             "    model,\n"
             "    finetune_vision_layers = False,\n"
             "    finetune_language_layers = True,\n"
             "    finetune_attention_modules = True,\n"
             "    finetune_mlp_modules = True,\n"
             "    r = 16, lora_alpha = 16, lora_dropout = 0, bias = 'none', random_state = 3407,\n"
             ")\n"
             "from unsloth.chat_templates import get_chat_template\n"
             "tokenizer = get_chat_template(tokenizer, chat_template = 'gemma-4')"),
    ("md", "## 2. Load the flashcards"),
    ("code", "from datasets import load_dataset\n"
             "data = load_dataset('json', data_files = {s: f'{DIR}/{s}.jsonl' for s in ['train', 'val', 'test']})\n"
             "def to_text(batch):\n"
             "    return {'text': [tokenizer.apply_chat_template(m, tokenize = False, add_generation_prompt = False)"
             ".removeprefix('<bos>') for m in batch['messages']]}\n"
             "train = data['train'].map(to_text, batched = True)\n"
             "val = data['val'].map(to_text, batched = True)\n"
             "print(train, val)\n"
             "print(train[0]['text'][:1500])"),
    ("md", "## 3. Answers from the untrained model (the 'before' picture)\nSaved now, so the head-to-head compares against the same starter brain."),
    ("code", "import json, torch\n"
             "test = data['test'].shuffle(seed = 7).select(range(3 if SMOKE_TEST else 40))\n"
             "MAX_NEW = 150 if SMOKE_TEST else 500\n"
             "def answer(messages, max_new_tokens = None):\n"
             "    max_new_tokens = max_new_tokens or MAX_NEW\n"
             "    prompt = [m for m in messages if m['role'] != 'assistant']\n"
             "    inputs = tokenizer.apply_chat_template(prompt, add_generation_prompt = True, tokenize = True,\n"
             "                                           return_dict = True, return_tensors = 'pt').to('cuda')\n"
             "    out = model.generate(**inputs, max_new_tokens = max_new_tokens, use_cache = True,\n"
             "                         temperature = 1.0, top_p = 0.95, top_k = 64)\n"
             "    text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens = True).strip()\n"
             "    return text.removeprefix('thought').strip()  # empty thinking-channel marker left by the template\n"
             "FastModel.for_inference(model)\n"
             "with model.disable_adapter():\n"
             "    before = [answer(r['messages']) for r in test]\n"
             "json.dump(before, open(f'{DIR}/answers_before.json', 'w'))\n"
             "print(before[0])"),
    ("md", "## 4. Train\nOne pass through every flashcard. Watch `loss` fall in the log. "
           "Checkpoints save to Drive every 200 steps."),
    ("code", "from trl import SFTTrainer, SFTConfig\n"
             "from unsloth.chat_templates import train_on_responses_only\n"
             "FastModel.for_training(model)\n"
             "trainer = SFTTrainer(\n"
             "    model = model, tokenizer = tokenizer,\n"
             "    train_dataset = train, eval_dataset = val,\n"
             "    args = SFTConfig(\n"
             "        dataset_text_field = 'text',\n"
             "        per_device_train_batch_size = 2,\n"
             "        gradient_accumulation_steps = 8,\n"
             "        warmup_ratio = 0.03,\n"
             "        num_train_epochs = 1,\n"
             "        max_steps = 10 if SMOKE_TEST else -1,\n"
             "        learning_rate = 1e-4,\n"
             "        lr_scheduler_type = 'cosine',\n"
             "        logging_steps = 10,\n"
             "        eval_strategy = 'steps', eval_steps = 5 if SMOKE_TEST else 200,\n"
             "        save_strategy = 'steps', save_steps = 5 if SMOKE_TEST else 200, save_total_limit = 2,\n"
             "        output_dir = f'{DIR}/checkpoints',\n"
             "        optim = 'adamw_8bit', weight_decay = 0.01, seed = 3407,\n"
             "        report_to = 'none',\n"
             "    ),\n"
             ")\n"
             "trainer = train_on_responses_only(trainer, instruction_part = '<|turn>user\\n', response_part = '<|turn>model\\n')\n"
             "import glob\n"
             "resume = bool(glob.glob(f'{DIR}/checkpoints/checkpoint-*'))\n"
             "stats = trainer.train(resume_from_checkpoint = resume)\n"
             "print(stats)"),
    ("md", "## 5. Save the trained add-on to Drive"),
    ("code", "model.save_pretrained(f'{DIR}/brand-manager-lora')\n"
             "tokenizer.save_pretrained(f'{DIR}/brand-manager-lora')\n"
             "print('saved')"),
    ("md", "## 6. Answers from the trained model (the 'after' picture)"),
    ("code", "FastModel.for_inference(model)\n"
             "after = [answer(r['messages']) for r in test]\n"
             "rows = [{'request': [m for m in r['messages'] if m['role'] == 'user'][0]['content'],\n"
             "         'reference': [m for m in r['messages'] if m['role'] == 'assistant'][0]['content'],\n"
             "         'before': b, 'after': a} for r, b, a in zip(test, before, after)]\n"
             "json.dump(rows, open(f'{DIR}/head_to_head.json', 'w'), indent = 2)\n"
             "for r in rows[:2]:\n"
             "    print('REQUEST:', r['request'][:400], '\\n\\nBEFORE:', r['before'][:600], '\\n\\nAFTER:', r['after'][:600], '\\n' + '=' * 80)"),
    ("md", "## 7. Try it yourself\nType your own brand question below and run the cell."),
    ("code", "question = 'We are launching a premium dog food at twice the price of the category leader. How do we justify the price?'\n"
             "print(answer([{'role': 'system', 'content': train[0]['messages'][0]['content']},\n"
             "              {'role': 'user', 'content': question}]))"),
]

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
                 "kernelspec": {"name": "python3", "display_name": "Python 3"}},
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": src} if kind == "md" else
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}
        for kind, src in cells
    ],
}
out = ROOT / "notebooks" / "train_brand_manager.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
