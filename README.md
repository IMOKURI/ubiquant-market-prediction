# Ubiquant Market Prediction

![image](https://user-images.githubusercontent.com/1638500/150118505-d05ff030-76b3-4418-9904-d9bb0b6123f0.png)


## Study

- [Perspectives on the comp evaluation metric and (potential) loss functions](https://www.kaggle.com/c/ubiquant-market-prediction/discussion/302874)
    - ピアソン相関係数が評価指標であることを踏まえた損失関数の検討
    - ピアソン相関係数はMSEなどとくらべて、外れ値があった場合や、値のシフト、スケールの違いなどの影響を受けにくい
    - 有力な損失関数
        - MSE
        - ピアソン相関係数 (PCC)
        - concordance correlation coefficient (CCC)
