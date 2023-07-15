# deep-speech

## Dataset
Librispeech: 1000 hours of read English speech with sampling rate of 16 kHz.

train split: "train-clean-100" has 28539 examples (585 chapters and 251 different speakers where each speakers reads unique sample of the chapters).
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

test split: "test-clean" has 2620 examples. (40 chapters read by 4 different speakers)

## Citation
https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch/
