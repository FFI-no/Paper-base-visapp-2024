###### Paper published at VISAPP 2024 - International Conference on Computer Vision Theory and Applications
### BASE: Probably a Better Approach to Visual Multi-Object Tracking
Authors: **M. Larsen**, S. Rolfsjord, D. Gusland, J. Ahlberg and K. Mathiassen (martin-vonheim.larsen (a) ffi.no)

&nbsp;[![License](images/badge-mit.svg)](https://opensource.org/license/mit) [![DOI](images/badge-doi.svg)](https://doi.org/10.5220/0012386600003660) [![arXiv](images/badge-arxiv.svg)](https://arxiv.org/abs/2309.12035)

The field of visual object tracking is dominated by methods that combine simple tracking algorithms and ad hoc schemes.  Probabilistic tracking algorithms, which are leading in other fields, are surprisingly absent from the leaderboards.  We found that accounting for distance in target kinematics, exploiting detector confidence and modelling non-uniform clutter characteristics is critical for a probabilistic tracker to work in visual track- ing.  Previous probabilistic methods fail to address most or all these aspects, which we believe is why they fall so far behind current state-of-the-art (SOTA) methods (there are no probabilistic trackers in the MOT17 top 100).  To rekindle progress among probabilistic approaches, we propose a set of pragmatic models addressing these challenges, and demonstrate how they can be incorporated into a probabilistic framework.  We present BASE (Bayesian Approximation Single-hypothesis Estimator), a simple, performant and easily extendible vi- sual tracker, achieving state-of-the-art (SOTA) on MOT17 and MOT20, without using Re-Id.

This repository provides the reference implementation of BASE, and reproduces the results presented in the paper on [MOT17](https://motchallenge.net/method/MOT=6175&chl=10) and [MOT20](https://motchallenge.net/method/MOT=6175&chl=13).

## Setup on Ubuntu 22.04
```bash
[sudo] apt install python3-pip python3-venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Estimate parameters from training set (optional)
```bash
python estimate_params.py
```
This takes several minutes, and should yield the following output:
```
--------------------------
Began estimation for MOT17
Loading data...
Estimating R...
R:
tensor([[ 6.5862,  0.1689, -0.9415, -0.1609],
        [ 0.1689,  8.5673,  0.3827,  6.0906],
        [-0.9415,  0.3827, 20.1033,  1.0013],
        [-0.1609,  6.0906,  1.0013, 38.7818]], dtype=torch.float64)
Loading some more data...
Estimating Pcr...
Pcr:
tensor([[3.0926, 0.0772],
        [0.0772, 1.1683]], dtype=torch.float64)
Generating p_w histogram... done
Generating P_C histgram... done
Performing MLE for process noise parameters...
sigma_ca: 0.71
sigma_sr: 7.85
wrote estimated params to ./data/mot17_estimated_params.pt
--------------------------
Began estimation for MOT20
Loading data...
Estimating R...
R:
tensor([[ 7.7281,  0.0696,  0.0872,  0.4300],
        [ 0.0696, 13.4898,  1.5277, 12.4674],
        [ 0.0872,  1.5277, 18.6464,  5.0624],
        [ 0.4300, 12.4674,  5.0624, 50.1261]], dtype=torch.float64)
Estimating Pcr...
Pcr:
tensor([[ 5.4781, -3.5756],
        [-3.5756,  6.1142]], dtype=torch.float64)
Generating p_w histogram... done
Generating P_C histgram... done
Performing MLE for process noise parameters...
sigma_ca: 0.23
sigma_sr: 2.67
wrote estimated params to ./data/mot20_estimated_params.pt
```
The resulting histograms and parameters are stored in
- `data/motXX_boxsize_hist.pt`
- `data/motXX_inlier_odds_hist.pt`
- `data/motXX_estimated_params.pt`

Parameters that have received additional manual tuning are provided in `data/motXX_manual_params.json`.

## Reproduce MOT17 results
```bash
python base_mot17.py
```

Which should yield the following output:
```
Processing sequence MOT17-01-FRCNN
Found no ground truth, running without metrics
took 3186.54ms
save results to ./data/results/mot17_base/MOT17-01-FRCNN.txt
Processing sequence MOT17-03-FRCNN
Found no ground truth, running without metrics
took 25919.08ms
save results to ./data/results/mot17_base/MOT17-03-FRCNN.txt
Processing sequence MOT17-06-FRCNN
Found no ground truth, running without metrics
took 9388.93ms
save results to ./data/results/mot17_base/MOT17-06-FRCNN.txt
Processing sequence MOT17-07-FRCNN
Found no ground truth, running without metrics
took 4838.90ms
save results to ./data/results/mot17_base/MOT17-07-FRCNN.txt
Processing sequence MOT17-08-FRCNN
Found no ground truth, running without metrics
took 6241.41ms
save results to ./data/results/mot17_base/MOT17-08-FRCNN.txt
Processing sequence MOT17-12-FRCNN
Found no ground truth, running without metrics
took 6630.63ms
save results to ./data/results/mot17_base/MOT17-12-FRCNN.txt
Processing sequence MOT17-14-FRCNN
Found no ground truth, running without metrics
took 6892.96ms
save results to ./data/results/mot17_base/MOT17-14-FRCNN.txt
done
```

Result files should now have been generated in `data/results/mot17_base`.
These can be compared to the files in `data/submission/mot17_base`, or by downloading the `FFI_BASE` submission from [motchallenge.net](https://motchallenge.net).

## Reproduce MOT20 results
```bash
python base_mot20.py
```
Which should yield the following output:
```
Processing sequence MOT20-04
Found no ground truth, running without metrics
took 117943.45ms
save results to ./data/results/mot20_base/MOT20-04.txt
Processing sequence MOT20-06
Found no ground truth, running without metrics
took 40749.37ms
save results to ./data/results/mot20_base/MOT20-06.txt
Processing sequence MOT20-07
Found no ground truth, running without metrics
took 7975.15ms
save results to ./data/results/mot20_base/MOT20-07.txt
Processing sequence MOT20-08
Found no ground truth, running without metrics
took 26335.79ms
save results to ./data/results/mot20_base/MOT20-08.txt
done
```

Result files should now have been generated in `data/results/mot20_base`.
These can be compared to the files in `data/submission/mot20_base`, or by downloading the `FFI_BASE` submission from [https://motchallenge.net](motchallenge.net).

## Citation
If you find the paper or code useful in your work, please cite as BibTeX
```bibtex
@conference{visapp24,
  author={Martin Larsen. and Sigmund Rolfsjord. and Daniel Gusland. and Jörgen Ahlberg. and Kim Mathiassen.},
  title={BASE: Probably a Better Approach to Visual Multi-Object Tracking},
  booktitle={Proceedings of the 19th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications - Volume 4: VISAPP},
  year={2024},
  pages={110-121},
  publisher={SciTePress},
  organization={INSTICC},
  doi={10.5220/0012386600003660},
  isbn={978-989-758-679-8},
  issn={2184-4321}
}
```
See [DOI:10.5220/0012386600003660](https://doi.org/10.5220/0012386600003660) for Harvard/EndNote citation.
