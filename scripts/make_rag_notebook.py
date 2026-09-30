"""Write notebooks/rag_brand_manager.ipynb: retrieval (RAG) on top of the fine-tuned model, plus a control.

Answers the same 40 held-out questions as head_to_head.json in two ways:
  tuned_rag = fine-tuned Gemma + retrieved playbook passages
  base_rag  = untrained Gemma + the same passages (control)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIR = "/content/drive/MyDrive/BrandManagerAI"

cells = [
    ("md", "# Brand Manager in a Box: add the playbook (RAG)\n\n"
           "**Before you start:** check the top right says A100, then Runtime > **Run all**.\n\n"
           "Takes about 1 to 1.5 hours. Keep this tab open. Results save to Drive `BrandManagerAI/rag_answers.json`."),
    ("code", "from google.colab import drive\n"
             "drive.mount('/content/drive')\n"
             f"DIR = '{DIR}'\n"
             "import os; print(sorted(os.listdir(DIR)))"),
    # Same fix as the training notebook: Gemma 4 12B needs transformers 5.10 (Unsloth pins <=5.5).
    ("code", "%%capture\n!pip install -U unsloth sentence-transformers\n!pip install 'transformers==5.10.2'"),
    ("md", "## 1. Build the search over the playbook\nEach passage becomes a list of numbers that captures its meaning. "
           "A question is turned into numbers the same way, and the closest passages are the best matches."),
    ("code", "import json, numpy as np, torch\n"
             "from sentence_transformers import SentenceTransformer\n"
             "passages = [json.loads(l) for l in open(f'{DIR}/passages.jsonl')]\n"
             "embedder = SentenceTransformer('BAAI/bge-base-en-v1.5', device = 'cuda')\n"
             "emb_path = f'{DIR}/passage_embeddings.npy'\n"
             "if os.path.exists(emb_path) and np.load(emb_path).shape[0] == len(passages):\n"
             "    P = np.load(emb_path)\n"
             "else:\n"
             "    P = embedder.encode([p['text'] for p in passages], batch_size = 128, normalize_embeddings = True,\n"
             "                        show_progress_bar = True)\n"
             "    np.save(emb_path, P)\n"
             "P = torch.tensor(P, device = 'cuda')\n"
             "QUERY = 'Represent this sentence for searching relevant passages: '\n"
             "def lookup(question, k = 4):\n"
             "    q = embedder.encode([QUERY + question], normalize_embeddings = True, convert_to_tensor = True)\n"
             "    top = torch.topk(P @ q[0], k).indices.tolist()\n"
             "    return [passages[i] for i in top]\n"
             "for p in lookup('How should I price a premium product above the category leader?'):\n"
             "    print(p['source'], '|', p['text'][:200], '\\n')"),
    ("md", "## 2. Load the trained bot (the add-on from the training run)"),
    ("code", "from unsloth import FastModel\n"
             "from unsloth.chat_templates import get_chat_template\n"
             "model, tokenizer = FastModel.from_pretrained(\n"
             "    model_name = f'{DIR}/brand-manager-lora',\n"
             "    max_seq_length = 4096,\n"
             "    load_in_4bit = False,\n"
             ")\n"
             "tokenizer = get_chat_template(tokenizer, chat_template = 'gemma-4')\n"
             "FastModel.for_inference(model)"),
    ("md", "## 3. Answer the 40 test questions, with the playbook open"),
    ("code", "SYSTEM = json.loads(open(f'{DIR}/test.jsonl').readline())['messages'][0]['content']\n"
             "def with_notes(question):\n"
             "    notes = '\\n\\n'.join(f'[{i + 1}] ' + p['text'] for i, p in enumerate(lookup(question)))\n"
             "    return (question + '\\n\\n---\\nNotes from your brand playbook. Use whatever helps and ignore the rest. '\n"
             "            'Never name the companies, brands, programs or people that appear in the notes.\\n\\n' + notes)\n"
             "tok = getattr(tokenizer, 'tokenizer', tokenizer)\n"
             "tok.padding_side = 'left'\n"
             "def clean(text):\n"
             "    return text.strip().removeprefix('thought').strip()  # empty thinking-channel marker\n"
             "def answer_batch(questions, max_new_tokens = 500):\n"
             "    convos = [[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': with_notes(q)}] for q in questions]\n"
             "    inputs = tokenizer.apply_chat_template(convos, add_generation_prompt = True, tokenize = True, padding = True,\n"
             "                                           return_dict = True, return_tensors = 'pt').to('cuda')\n"
             "    out = model.generate(**inputs, max_new_tokens = max_new_tokens, use_cache = True,\n"
             "                         temperature = 1.0, top_p = 0.95, top_k = 64)\n"
             "    n = inputs['input_ids'].shape[1]\n"
             "    return [clean(tokenizer.decode(o[n:], skip_special_tokens = True)) for o in out]\n"
             "def answer_all(questions, size = 8):\n"
             "    results = []\n"
             "    for i in range(0, len(questions), size):\n"
             "        results += answer_batch(questions[i:i + size])\n"
             "        print(f'{len(results)} / {len(questions)} done')\n"
             "    return results\n"
             "rows = json.load(open(f'{DIR}/head_to_head.json'))\n"
             "questions = [r['request'] for r in rows]\n"
             "tuned_rag = answer_all(questions)\n"
             "with model.disable_adapter():\n"
             "    base_rag = answer_all(questions)\n"
             "for r, t, b in zip(rows, tuned_rag, base_rag):\n"
             "    r['tuned_rag'], r['base_rag'] = t, b\n"
             "json.dump(rows, open(f'{DIR}/rag_answers.json', 'w'), indent = 2)\n"
             "print('saved', len(rows))\n"
             "print('\\nQUESTION:', rows[0]['request'][:300], '\\n\\nTRAINED + PLAYBOOK:', tuned_rag[0][:900])"),
    ("md", "## 4. Try it yourself"),
    ("code", "question = 'We are launching a premium dog food at twice the price of the category leader. How do we justify the price?'\n"
             "print(answer_batch([question])[0])"),
    ("md", "## 5. Switch off the GPU\nWhen you've finished: Runtime > **Disconnect and delete runtime** > Yes."),
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
out = ROOT / "notebooks" / "rag_brand_manager.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
