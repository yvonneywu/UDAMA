### 📖 UDAMA: Unsupervised Domain Adaptation through Multi-discriminator Adversarial Training with Noisy Labels Improves Cardio-fitness Prediction

This repository contains the implementation code for the paper:

![header image](https://github.com/dengerrrr/UDAMA-CRF-Prediction/blob/main/udama_mlhc.png)

## Abstract
Deep learning models have shown great promise in various healthcare monitoring applications. However, most healthcare datasets with high-quality (gold-standard) labels are
small-scale, as directly collecting ground truth is often costly and time-consuming. As a result, models developed and validated on small-scale datasets often suffer from overfitting
and do not generalize well to unseen scenarios. At the same time, large amounts of imprecise (silver-standard) labeled data, annotated by approximate methods with the help of
modern wearables and in the absence of ground truth validation, are starting to emerge. However, due to measurement differences, this data displays significant label distribution
shifts, which motivates the use of domain adaptation. To this end, we introduce UDAMA, a method with two key components: Unsupervised Domain Adaptation and Multidiscriminator Adversarial Training, where we pre-train on the silver-standard data and employ adversarial adaptation with the gold-standard data along with two domain discriminators. In particular, we showcase the practical potential of UDAMA by applying it to Cardio-respiratory fitness (CRF) prediction. CRF is a crucial determinant of metabolic disease and mortality, and it presents labels with various levels of noise (goldand silver-standard), making it challenging to establish an accurate prediction model. Our results show promising performance by alleviating distribution shifts in various label shift settings. Additionally, by using data from two free-living cohort studies (Fenland and BBVS), we show that UDAMA consistently outperforms up to 12% compared to competitive transfer learning and state-of-the-art domain adaptation models, paving the way for
leveraging noisy labeled data to improve fitness estimation at scale.

## Data 
We use data from the [Fenland Study](https://www.mrc-epid.cam.ac.uk/research/studies/fenland/) and the [Biobank Validation Study](https://www.mrc-epid.cam.ac.uk/research/studies/uk-biobank-validation/). We cannot publicly share this data, but it is available from the MRC Epidemiology Unit at the University of Cambridge upon reasonable request.



## Updates
The code will be released soon. 


## Citation 
Yu Wu, Dimitris Spathis, Hong Jia, Ignacio Perez-Pozuelo, Tomas Gonzales, Soren Brage, Nicholas Wareham, Cecilia Mascolo. ["UDAMA: Unsupervised Domain Adaptation through Multi-discriminator Adversarial Training with Noisy Labels Improves Cardio-fitness Prediction"](https://arxiv.org/abs/2307.16651)
In Machine Learning for Healthcare Conference. New York, USA, 2023



