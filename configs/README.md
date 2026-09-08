# configs

하이퍼파라미터를 코드에서 분리한다. (README 18번)

값을 코드에 직접 적으면 실험할 때마다 코드를 고쳐야 하고,
어떤 값으로 어떤 결과가 나왔는지 추적할 수 없다.

```yaml
model:
  hidden_size: 512
  layers: 8
  heads: 8

training:
  learning_rate: 0.0005
  batch_size: 32

lora:
  rank: 16
  alpha: 32
```

실험마다 config 파일을 따로 두고, `experiments/`의 기록에서 그 파일명을 참조한다.
