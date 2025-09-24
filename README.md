## This is the code repository for the paper EReLiFM: Evidential Reliability-Aware Residual Flow Meta-Learning for Open-Set Domain Generalization under Noisy Labels


>Abstract: Open-Set Domain Generalization (OSDG) aims to enable deep learning models to recognize unseen categories in new domains, which is crucial for real-world applications. Label noise hinders open-set domain generalization by corrupting source-domain knowledge, making it harder to recognize known classes and reject unseen ones. While existing methods address OSDG under Noisy Labels (OSDG-NL) using hyperbolic prototype-guided meta-learning, they struggle to bridge domain gaps, especially with limited clean labeled data. In this paper, we propose Evidential Reliability-Aware Residual Flow Meta-Learning (EReLiFM). We first introduce an unsupervised two-stage evidential loss clustering method to promote label reliability awareness. Then, we propose a residual flow matching mechanism that models structured domain- and category-conditioned residuals, enabling diverse and uncertainty-aware transfer paths beyond interpolation-based augmentation. During this meta-learning process, the model is optimized such that the update direction on the clean set maximizes the loss decrease on the noisy set, using pseudo labels derived from the most confident predicted class for supervision. Experimental results show that EReLiFM outperforms existing methods on OSDG-NL, achieving state-of-the-art performance.

### 1. Dataset paths

PACS dataset and DigitsDG dataset can be obtained from their own websites. Links will be provided after the anonymous submission stage. 

```
# PACS
Known classes: ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house']
Unknown classes: ['person']
```

### 2. Dataset Construction

The dataset needs to be divided into two folders for training and validation. We provide reference code for automatically dividing data using official split in `data_list/split_kfold.py`.

```python
root_dir = "path/to/PACS"
instr_dir = "path/to/PACS_data_list"
```

### 3. Train

To run the training code, please update the path of the dataset in `ml_open.py`:

```python
if dataset == 'PACS':	
    train_dir = 'path/to/PACS_train' # the folder of training data 
	val_dir = 'path/to/PACS_val' # the folder of validation data 
	test_dir = 'path/to/PACS_all' or ['path/to/PACS_train', 'path/to/PACS_val']
```

then simply run:

```python
python train_file.py
```

### 4. Folder Structure

the training dynamics can be obatined via running the train_loss_recording.py code under residual_flow_meta_learning/ folder, and then we run UTS-ELC.py to achieve clean/noisy partition. After that we train residual flow matching model according to the instructions in DC-CRFM folder, and then run train_EReLiFM.py in residual_flow_meta_learning/ folder.

