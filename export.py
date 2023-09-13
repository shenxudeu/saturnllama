import os
import sys
import argparse
import json
from pathlib import Path
import torch
from torch import nn

from model import ModelArgs, Transformer

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def load_llama(llama_path):
    """
    Load a model from a llama directory.
    """
    params_path = os.path.join(llama_path, "params.json")
    with open(params_path, "r") as f:
        params = json.load(f)
        print(params)

    model_paths = sorted(list(Path(llama_path).glob('consolidated.*.pth')))
    models = [torch.load(p, map_location='cpu') for p in model_paths]

    def concat_weights(models):
        state_dict = {}
        for name in list(models[0]):
            tensors = [model[name] for model in models]
            if len(tensors) == 1 or len(tensors[0].shape) == 1:
                state_dict[name] = tensors[0]
                continue
            is_axis_1 = (
                name.startswith('tok_embeddings.')
                or name.endswith('.attention.wo.weight')
                or name.endswith('.feed_forward.w2.weight')
            )
            axis = 1 if is_axis_1 else 0
            state_dict[name] = torch.cat(tensors, dim=axis)
            for model in models:
                del model[name]
        return state_dict

    state_dict = concat_weights(models)
    del models

    # DEBUG
    # params["dim"] = 512
    # params["n_layers"] = 6
    # params["n_heads"] = 8

    # set up model args
    args = ModelArgs()
    args.flash = True
    args.dim = params["dim"]
    args.n_layers = params["n_layers"]
    args.n_heads = params["n_heads"]
    args.n_kv_heads = params.get("n_kv_heads") or params["n_heads"]
    args.multiple_of = params["multiple_of"]
    args.norm_eps = params["norm_eps"]

    model = Transformer(args)

    print(f"Total number of parameters: {count_parameters(model) / 1e9} Billion")

    model.tok_emb = nn.Parameter(state_dict["tok_embeddings.weight"])
    model.norm.weight = nn.Parameter(state_dict["norm.weight"])

    for layer in model.layers:
        i = layer.layer_id
        layer.attention_norm.weight = nn.Parameter(state_dict[f"layers.{i}.attention_norm.weight"])
        layer.attention.wq.weight = nn.Parameter(state_dict[f"layers.{i}.attention.wq.weight"])
        layer.attention.wk.weight = nn.Parameter(state_dict[f"layers.{i}.attention.wk.weight"])
        layer.attention.wv.weight = nn.Parameter(state_dict[f"layers.{i}.attention.wv.weight"])
        layer.attention.wo.weight = nn.Parameter(state_dict[f"layers.{i}.attention.wo.weight"])
        layer.ffn_norm.weight = nn.Parameter(state_dict[f"layers.{i}.ffn_norm.weight"])
        layer.feed_forward.w1.weight = nn.Parameter(state_dict[f"layers.{i}.feed_forward.w1.weight"])
        layer.feed_forward.w2.weight = nn.Parameter(state_dict[f"layers.{i}.feed_forward.w2.weight"])
        layer.feed_forward.w3.weight = nn.Parameter(state_dict[f"layers.{i}.feed_forward.w3.weight"])

    model.output.weight = nn.Parameter(state_dict["output.weight"])
    model.eval()

    print(f"{llama_path} model loaded. {len(state_dict)} sets of parameters loaded.")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama", type=str, default=None)
    args = parser.parse_args()

    model = load_llama(args.llama)
