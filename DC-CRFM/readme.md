
# Official implementation of "DC-CRFM"



> Abstract: DC-CRFM (Domain and Category Conditioned Residual Flow Matching) is a flow-matching strategy proposed in EReLiFM for open-set domain generalization under noisy labels. Instead of using interpolation-based augmentation like MixUp, DC-CRFM explicitly learns structured residuals conditioned on both domain and category labels, capturing meaningful transitions across categories and domains. It generates domain residuals to model visual differences of the same category across domains and category residuals to capture discrepancies between categories within the same domain. By enriching transfer paths with diverse and structured variations, DC-CRFM enhances cross-domain generalization and proves fundamentally more effective than interpolation-based methods.






## Installation

Python `3.10` and Pytorch `1.13.1`/`2.0.0` are used in this implementation.
Please install required libraries:

```
pip install -r requirements.txt
```


## Training

All training scripts are wrapped in [run.sh](bash_scripts/run.sh). Simply comment/uncomment the relevant commands and run `bash run.sh`.



