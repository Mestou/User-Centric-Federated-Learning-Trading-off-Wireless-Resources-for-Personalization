# User_centric_Federated_Learning: Trading off Wireless Resources for Personalization
Tensorflow implementation of the algorithm "PER" implemented in the paper titled ["User-Centric Federated Learning: Trading off Wireless Resources for Personalization"](https://arxiv.org/abs/2304.12930) <br/>



## Environment setup:
Packages can be found in `requirements.txt`.

## Training:

The list of the available Algorithms [PER,Fedprox,Scaffold,Ditto,VAN,LOC,pFedME] with their parameters, and the available models [CNN,VGG,Sentiment] are included (commented) in the `params.yaml` file.

We include code to run experiments on CIFAR-10 (CNN or VGG models) and Stack-Overflow Sentiment (Sentiment model) and EMNIST (CNN or VGG models) as in the paper. Choose the corresponding algorithm, dataset and arguments in the `params.yaml` file. Then, to train the model, simply run:

```
python3 main.py
```

## Results
Results are populated in the Results folder.

## References
If you find this work useful in your research, please consider citing one of our related papers:
```

@INPROCEEDINGS{9682003,
  author={Mestoukirdi, Mohamad and Zecchin, Matteo and Gesbert, David and Li, Qianrui and Gresset, Nicolas},
  booktitle={2021 IEEE Globecom Workshops (GC Wkshps)}, 
  title={User-Centric Federated Learning}, 
  year={2021},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/GCWkshps52748.2021.9682003}}

@ARTICLE{10286560,
  author={Mestoukirdi, Mohamad and Zecchin, Matteo and Gesbert, David and Li, Qianrui},
  journal={IEEE Transactions on Machine Learning in Communications and Networking}, 
  title={User-Centric Federated Learning: Trading off Wireless Resources for Personalization}, 
  year={2023},
  volume={1},
  number={},
  pages={346-359},
  doi={10.1109/TMLCN.2023.3325297}}
