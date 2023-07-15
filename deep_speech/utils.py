import json
import torch
import torch.nn as nn

class TextTransform:
    """Maps characters to integers and vice versa"""

    def __init__(self):
        self.char_map_str = open("../char_map.txt", "r").read()
        self.char_map = {}
        self.index_map = {}
        for line in self.char_map_str.strip().split("\n"):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = " "

    def text_to_int(self, text):
        """Use a character map and convert text to an integer sequence"""
        int_sequence = []
        for c in text:
            if c == " ":
                ch = self.char_map["<SPACE>"]
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """Use a character map and convert int labels to an text sequence"""
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return "".join(string).replace("", " ")

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

def GreedyDecoder(output, labels, label_lengths, text_transform, blank_label=28, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j!=0 and index == args[j - 1]: # the current index is the same as the previous index
                    continue 
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
    return decodes, targets
