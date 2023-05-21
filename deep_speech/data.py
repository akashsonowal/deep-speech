import json


class TextTransform:
    """Maps characters to integers and vice versa"""

    def __init__(self):
        self.char_map_str = json.load(open("char_map.json", "r"))
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split("\n"):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = " "

    def text_to_int(self, text):
        """Use a character map and convert text to an integer sequence"""
        int_sequence = []
        for c in text:
            if c == " ":
                ch = self.char_map[""]
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

def data_processing(data, data_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    # (tensor([[-0.0096, -0.0099, -0.0088,  ...,  0.0024,  0.0026,  0.0027]]), 16000, "PORTHOS FANCIES HIMSELF AT LA ROCHELLE THOUGHT D'ARTAGNAN AS THEY RETURNED FREIGHTED WITH BOTTLES", 8063, 274116, 36)
    for (waveform, _, utterance, _, _, _) in data:
        if data_type == "train":
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
