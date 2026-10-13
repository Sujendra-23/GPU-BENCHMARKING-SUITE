# ResNet TensorRT quantization

| model | precision | top-1 % | delta pp vs FP32 | median ms | speedup | engine MiB | size / FP32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| resnet18 | fp32 | 59.38 | 0.00 | 0.504 | 1.00 | 73.79 | 1.00 |
| resnet18 | fp16 | 59.38 | 0.00 | 0.190 | 2.65 | 23.47 | 0.32 |
| resnet18 | int8 | 60.94 | 1.56 | 0.186 | 2.72 | 12.57 | 0.17 |
| resnet50 | fp32 | 82.03 | 0.00 | 0.991 | 1.00 | 124.36 | 1.00 |
| resnet50 | fp16 | 82.03 | 0.00 | 0.353 | 2.81 | 50.27 | 0.40 |
| resnet50 | int8 | 81.25 | -0.78 | 0.364 | 2.72 | 27.17 | 0.22 |
