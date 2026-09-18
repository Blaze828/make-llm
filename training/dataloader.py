import torch


def pack_documents(documents, context_length, pad_id=0):
    """Pack tokenized documents; long documents overlap by one token between chunks."""
    if context_length < 2:
        raise ValueError("context_length must be >= 2")
    rows, ids, docs = [], [], []
    def flush():
        if not ids:
            return
        n = len(ids)
        rows.append({"input_ids": torch.tensor(ids+[pad_id]*(context_length-n)),
                     "document_ids": torch.tensor(docs+[-1]*(context_length-n)),
                     "attention_mask": torch.tensor([True]*n+[False]*(context_length-n))})
        ids.clear()
        docs.clear()
    for number, document in enumerate(documents):
        document = list(document)
        if len(document) < 2:
            continue
        if len(ids)+len(document) > context_length:
            flush()
        while len(document) > context_length:
            ids.extend(document[:context_length]); docs.extend([number]*context_length)
            flush()
            document = document[context_length-1:]
        ids.extend(document); docs.extend([number]*len(document))
        if len(ids) == context_length:
            flush()
    flush()
    return rows


def collate(rows, device):
    return {key: torch.stack([row[key] for row in rows]).to(device) for key in rows[0]}
