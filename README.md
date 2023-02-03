# Probable Domain Generalization via Quantile Risk Minimization
Code for reproducing the results in our NeurIPS 2022 paper [Probable Domain Generalization via Quantile Risk Minimization](https://arxiv.org/abs/2207.09944).

<p align="center">
  <img src="https://github.com/cianeastwood/qrm/blob/clean/assets/overview_qrm.png?raw=true" width="500" alt="QRM" />
</p>


## Repo structure
This repo contains four independent sub-repos which correspond to the experiments of our paper, 
namely: 
1. LinearRegression
2. CMNIST
3. WILDS
4. DomainBed

Each sub-repo contains its own README with instructions on 
how to get set up and reproduce experiments. We chose to keep these repos separate in order to allow users to:
- Make use of existing benchmark code, in particular, that of 
[WILDS](https://github.com/p-lambda/wilds/) and [DomainBed](https://github.com/facebookresearch/DomainBed).
- Only concern themselves with sub-repos of interest and their installation requirements.


## Contact
- For queries regarding sub-repos 1, 2 and 4, contact or tag [@cianeastwood](https://www.github.com/cianeastwood). 
- For queries regarding sub-repo 3, contact or tag [@arobey1](https://www.github.com/arobey1). 

## BibTex

```
@inproceedings{eastwood2022probable,
    title={Probable Domain Generalization via Quantile Risk Minimization},
    author={Eastwood, Cian and Robey, Alexander and Singh, Shashank and von K\"ugelgen, Julius and Hassani, Hamed and Pappas, George J. and Sch\"olkopf, Bernhard},
    booktitle={Advances in Neural Information Processing Systems},
    year={2022}
}
```
