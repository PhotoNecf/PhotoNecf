# Web Photo Source Identification based on Neural Enhanced Camera Fingerprint

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


## Next-step
The following content will also be released after paper acceptance.
- [ ] Release of benchmark dataset containing 1665 RAW photos.
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
