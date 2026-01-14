# VL-JEPA 
 This VL-JEPA implimentation takes direct insperation from the original [VL-JEPA paper](https://arxiv.org/pdf/2512.10942)

---
## Components Breakdown

1. X-Encoder (Video Encoder)
    I used a frozen DINOv3 VIT-S model 
2. QueryEmbedding (Used the embedding layers of `google/gemma-3-270m-it`)
3. The predictor is also `google/gemma-1-270m-it` where I took the last 4 layers of the model
    The first and last layers without the embedding layer
4. Y-Encoder is the same `EmbeddingGemma` model from the paper. Which I have also frozen 

> The goal is to train a good predictor for the two pretrained models (DINOv3 & EmbeddingGemma)

---
### My Asumptions 
1. I assume the pretrained models already have a unique representation of the world
2. Each model has a different internal representation 

[x] I used projection layers to help map from one models representaion to the others' representation
[x] This also comes in handy to map different model output dimentions [D] to each other 
