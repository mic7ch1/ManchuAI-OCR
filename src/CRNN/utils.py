import torch


def greedy_decode(logits, idx2char):
    seqs = logits.argmax(2).cpu().numpy().T
    out = []
    for s in seqs:
        last = None
        chars = []
        for i in s:
            if i != 0 and i != last:
                chars.append(idx2char.get(int(i), ""))
            last = i
        out.append("".join(chars))
    return out


def build_character_dict(datasets, text_key="manchu"):
    char2idx = {"<blank>": 0, "<unk>": 1}
    idx2char = {0: "<blank>", 1: "<unk>"}

    chars = set()
    for dataset in datasets:
        for sample in dataset:
            text = sample[text_key]
            chars.update(text)

    for i, ch in enumerate(sorted(chars)):
        char2idx[ch] = i + 2
        idx2char[i + 2] = ch

    return char2idx, idx2char


def collate_crnn_batch(batch):
    imgs, lbls, lens, raws = zip(*batch)
    return (torch.stack(imgs), torch.cat(lbls), torch.tensor(lens), list(raws))
