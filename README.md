# SelANet: Decision-Assisting Selective Sleep Apnea Detection Based on Confidence Score

## Overview

SelANet is an algorithm developed to detect sleep apnea using limited biological signals (oxygen saturation and ECG-derived respiration) while minimizing uncertainty in AI judgments. The algorithm employs selective prediction based on confidence scores to classify only high-confidence samples and reject uncertain ones, providing a second opinion to clinicians.

## Key Features

- Utilizes autoencoder for feature extraction from latent vectors
- Employs 1D CNN-LSTM architecture for confidence score measurement
- Implements a selection function based on target coverage
- Achieves improved classification performance through selective prediction

## Dataset

- Polysomnography data from 994 subjects obtained from Massachusetts General Hospital

## Performance

- Coverage violation: average of 0.067
- Classification performance:
  - Accuracy: 90.26%
  - Sensitivity: 91.29%
  - Specificity: 89.21%
- Improvement of approximately 7.03% in all metrics compared to non-selective prediction

## Advantages

1. Minimizes false diagnoses in cases of high uncertainty
2. Applicable to wearable devices despite low signal quality
3. Can be used as a simple detection method to determine the need for polysomnography or complement it

## Model Architecture

### TCN-based Autoencoder
<img src="https://media.springernature.com/full/springer-static/image/art%3A10.1186%2Fs12911-023-02292-3/MediaObjects/12911_2023_2292_Fig1_HTML.png?as=webp" alt="TCN-based Autoencoder" width="80%">

### Selective Prediction Model
![Selective Prediction Model](https://media.springernature.com/full/springer-static/image/art%3A10.1186%2Fs12911-023-02292-3/MediaObjects/12911_2023_2292_Fig2_HTML.png?as=webp){width=50%}

### TSNE Result
![TSNE Result](https://media.springernature.com/lw685/springer-static/image/art%3A10.1186%2Fs12911-023-02292-3/MediaObjects/12911_2023_2292_Fig5_HTML.png?as=webp){width=50%}


## Conclusion

SelANet demonstrates the effectiveness of selective prediction in sleep apnea detection using limited biological signals. It offers a promising approach for real-time sleep monitoring and personalized sleep quality assessment through wearable devices.

## Citation

Bark, B., Nam, B. & Kim, I.Y. SelANet: decision-assisting selective sleep apnea detection based on confidence score. BMC Med Inform Decis Mak 23, 190 (2023). https://doi.org/10.1186/s12911-023-02292-3