# deep-speech

## Dataset
Librispeech: 1000 hours of read English speech with sampling rate of 16 kHz.

train split: "train-clean-100" has 28539 examples (251 speakers)
```
- speaker_id
  - chapter id
    - speaker_id_chapter_id_000.flac
    - speaker_id_chapter_id_000.flac
    - ...
    - speaker_id_chapter_id.trans.txt
- speaker_id
...  
```
test split: "test-clean" has 2620 examples. (40 speakers)

## Experiment
```
python3 experiment.py
```

## Citation
https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch/
