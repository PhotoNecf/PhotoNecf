# Web Photo Source Identification based on Neural Enhanced Camera Fingerprint

## Updates!!
- 【2022-12-22】: We supplement the experimental results for evaluating the security of spatial splitting. We also supplement the results of our released model on a supplementary test dataset (1,276 RAW photos from 15 Android smartphone cameras) and a public JPEG-based dataset (34,427 JPEG photos from 35 devices), both with good performance. We will release our Android dataset after acceptance of the paper.

## Introduction
Source camera identification of web photos aims to establish a reliable linkage from the captured images to their source cameras, and has a broad range of applications, such as image copyright protection, user authentication, investigated evidence verification, etc. 
Our paper presents an innovative and practical source identification framework that employs neural-network enhanced sensor pattern noise to trace back web photos efficiently while ensuring security. 
The codes for fingerprint extraction network and benchmark dataset with modern smartphone cameras photos are all publicly available in this repo.

## Preparations
* Download the fingerprint extraction network model file and put them under directory `models/ckpts/`.
* Download the RAW photos (about 17G, will be released after paper acceptance) and put them in `data/images/`. 
* Install python requirements in `requirements.txt`.

## Run and Evaluation
Run fingerprint extraction network and save fingerprints in `results/fingerprints`:
```bash
python3 run_extract.py
```

Then run evaluation scripts on above fingerprint file as:
```bash
python3 run_ncc.py
```
You should see the output performance. In the case of single photo registration, the result is
```
- Results locates at：/path/PhotoNecf/results/measures/fingerprints_vs_fingerprints.npy, AUC: 99.80265567168468EER: 1.6559416559416558
```

## Supplementary Experimental Results

### Security Evaluation of Spatial Splitting

For evaluating the security of spatial splitting, we first derive one fingerprint from each RAW odd photo and one fingerprint from each RAW even photo for 15 iPhone cameras (i.e., resulting in two sets of 1,665 fingerprints) based on our released model under multiple registration setting with N=40. Then for each camera, we calculate AUC from the correlation matrix between its odd fingerprints and even fingerprints.

The following figure illustrates the correlation matrix and AUC for each camera. We got an average of 96.22% AUC with 5.33% standard deviation, which indicates relatively low information leakage.

<figure>
    <div style="text-align: center;">
        <img src=./odd_even_fp_corr_mat.jpeg width=80% />
    </div>
</figure>



Moreover, we calculated AUC and EER from the correlation matrix between RAW odd photos and the correlation matrix between RAW even photos, i.e., two 1665X1665 matrices . The results show 99.99% AUC and 0.253% EER for RAW odd photos, and 99.92% AUC and 0.497% EER for RAW even photos, both indicating highly discriminative ability.

### Network Performance on Android RAW photos and JPEG photos

While our released model was trained only on iPhone RAW photos, it displayed superior generalization and adaptability on both RAW Android photos and JPEG compressed photos.

For examining Android RAW photos, we provide a test dataset with 1,276 RAW photos from 15 Android smartphone cameras which will be released after paper acceptance. The following table shows the fingerprint accuracy performance comparison of our algorithm with previous algorithms on this dataset. The results with * indicate containing post-processing (ZM & WF). As shown in the table, our model outperforms conventional algorithms by a large margin with much higher AUC and lower EER.

<figure>
    <div style="text-align: center;">
        <img src=./results_android_photos.jpeg width=50% />
    </div>
</figure>

For examining JPEG compressed photos, we directly tested our released model on the VISION dataset[^1] (35 devices with 34,427 JPEG photos). On this JPEG compressed dataset we obtained 92.83% AUC, indicating better discrimination than other SOTA methods.

## Next-step
The following content will also be released after paper acceptance.
- [ ] Release of two benchmark datasets: a dataset containing 1,665 RAW photos from 15 iPhone cameras and a newly supplementary dataset containing 1,276 RAW photos from 15 Android smartphone cameras.
- [ ] Release of Training codes for fingerprint extraction network.
- [ ] Integration codes (including cryptographic schemes).


## License
The code is released under MIT license

```bash
MIT License

Copyright (c) 2022 PhotoNecf

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

[^1]: Shullani, D., Fontani, M., Iuliani, M. et al. VISION: a video and image dataset for source identification. EURASIP J. on Info. Security 2017, 15 (2017).
