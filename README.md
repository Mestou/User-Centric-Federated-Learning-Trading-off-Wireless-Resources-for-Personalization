# User_centric_Federated_Learning: Trading off Wireless Resources for Personalization
Tensorflow implementation for the paper titled "User-Centric Federated Learning: Trading off Wireless Resources for Personalization"

["User-Centric Federated Learning: Trading off Wireless Resources for Personalization"]([https://arxiv.org/pdf/2209.15328.pdf](https://arxiv.org/abs/2304.12930)) <br/>


## Environment setup:
Packages can be found in `requirements.txt`.

## Training:
Set the `params_path` in `main.py` to the the path of the `{}.yaml` file with the desired algorithm, model, and datasets. The default parameters can be found in the provided `{}.yaml` files. To train the model, simply run:

The list of the available Algorithms with their parameters, and the available models are included (commented) in the params.yml

```
python3 main.py
```
## Results
Results are populated in the Results folder and can be plotted using the plot.py script.

## References
If you find this work useful in your research, please consider citing one of our related papers:
```

@misc{mestoukirdi2023usercentric,
      title={User-Centric Federated Learning: Trading off Wireless Resources for Personalization}, 
      author={Mohamad Mestoukirdi and Matteo Zecchin and David Gesbert and Qianrui Li},
      year={2023},
      eprint={2304.12930},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}


@article{DBLP:journals/corr/abs-2110-09869,
  author       = {Mohamad Mestoukirdi and
                  Matteo Zecchin and
                  David Gesbert and
                  Qianrui Li and
                  Nicolas Gresset},
  title        = {User-Centric Federated Learning},
  journal      = {CoRR},
  volume       = {abs/2110.09869},
  year         = {2021},
  url          = {https://arxiv.org/abs/2110.09869},
  eprinttype    = {arXiv},
  eprint       = {2110.09869},
  timestamp    = {Mon, 25 Oct 2021 20:07:12 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2110-09869.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
