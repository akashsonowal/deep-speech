import json
import torch
import torch.nn as nn





def data_processing(data, audio_transforms, text_transform):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    # (tensor([[-0.0096, -0.0099, -0.0088,  ...,  0.0024,  0.0026,  0.0027]]), 16000, "PORTHOS FANCIES HIMSELF AT LA ROCHELLE THOUGHT D'ARTAGNAN AS THEY RETURNED FREIGHTED WITH BOTTLES", 8063, 274116, 36)
    for waveform, _, utterance, _, _, _ in data:
        spec = audio_transforms(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0] // 2)
        label_lengths.append(len(label))

    spectrograms = (
        nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
        .unsqueeze(1) #(batch_size, 1, max_time_steps, feature_dim) 
        .transpose(2, 3) #(batch_size, 1, feature_dim, max_time_steps) 
    ) 
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True) #(batch_size, max_label_length)
    return spectrograms, labels, input_lengths, label_lengths
