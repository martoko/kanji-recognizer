import argparse

import torch

import kanji
from recognizer.model import KanjiRecognizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a checkpoint to torchscript.")
    parser.add_argument('input', metavar='INPUT', type=str,
                        help="load checkpoint from this path")
    parser.add_argument('-o', '--output', type=str, default="traced_torchscript_model.pt",
                        help="save model to this path")
    args = parser.parse_args()

    model = KanjiRecognizer(len(kanji.frequent_kanji_plus))
    model.load_state_dict(torch.load(args.input)["model_state_dict"])

    x = torch.rand(1, 3, 32, 32)
    traced_model = torch.jit.trace(model, x)

    torch.jit.save(traced_model, args.output)
