# TFRecord Example

## # Description
- "Generate_TFRecord.py" is to generate tfrecords from flowers dataset.
- "Read_TFRecord.py" is to read tfrecords (tf.data API)
- "Read_Threading.py" is to read tfrecords (Threading)

## # Experiments
- batch size is 64.
- Weakly Augmentation is flip and random crop.
- Strongly Augmentation is RandAugment.

| Preprocessing | tf.data API (ms/64 images) | Threading API (ms/64 images) |
| ------------- | ---------------- | ------------------ |
| No Augmentation | 20~30ms | 20~30ms |
| Weakly Augmentation | 30~40ms | 40~50ms|
| Strongly Augmentation | 2400~2500ms | 4300~4600ms |

