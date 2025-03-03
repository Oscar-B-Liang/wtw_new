# Go1 Sim-to-Real Locomotion

## Overview <a name="overview"></a>

This repository provides an implementation of the paper:

<td style="padding:20px;width:75%;vertical-align:middle">
      <a href="https://sites.google.com/berkeley.edu/efficient-locomotion" target="_blank">
      <b> Adaptive Energy Regularization for Autonomous Gait Transition and Energy-Efficient Quadruped Locomotion </b>
      </a>
      <br>
      Boyuan Liang, Lingfeng Sun, Xinghao Zhu, Bike Zhang, Yixiao Wang, Ziyin Xiong, Chenran Li, Koushil Sreenath and Masayoshi Tomizuka
      <br>
      <br>
      <a href="https://arxiv.org/abs/2403.20001">paper</a> /
      <a href="https://sites.google.com/berkeley.edu/efficient-locomotion" target="_blank">project page</a>
    <br>
</td>

<br>

## Usage <a name="Usage"></a>
* Train a policy: `python scripts/train.py --headless --device 0 --seed 0 --cfg adaptive_en --en_new_actual 1.0`
  + --headless: Weather to pop out renderer of isaacgym while training. Usually preferred not to pop them out to save PC resources.
  + --device DEVICE: If you have multiple GPUs on your PC, you can specify which GPU to use.
  + --seed SEED: Randomization seed in Pytorch.
  + --cfg CFG: There are three configuration files in `go1_gym/envs/go1`, each corresponding to a different training environment. `adaptive_en` uses the configuration `go1_gym/envs/go1/go1_config_adaptive.py`, which is training on a flat ground. `adaen_terrain` uses the configuration `go1_gym/envs/go1/go1_config_adaptive_terrain.py`, which is training on terrains.
  + --en_new_actual EN: How much energy regularization is added, corresponding to the value $\alpha_{en}$. Under `adaptive_en` and `adaen_terrain` configurations, values between $0.8$ and $1.0$ should in general work.

After training, a folder called `checkpoint` will be created, and inside it will store the trained models.
* Play a trained policy: `python scripts/play.py --device 0 --headless --lin_speed 2.0 --ang_speed 0.0 --terrain_choice flat --terrain_diff 0.1 --model_dir checkpoints/train/seed-10-ennewa-1.0-ennewc-0.0`
  + --headless: Weather to pop out renderer of isaacgym while training. Usually preferred not to pop them out to save PC resources. Do not worry, the play script will record a video for the simulation so you can watch remotely.
  + --device DEVICE: If you have multiple GPUs on your PC, you can specify which GPU to use.
  + --lin_speed SPEED: target linear speed you want the robot to move.
  + --ang_speed SPEED: target angular speed you want the robot to move.
  + --terrain_choice CHOICE: if the policy is trained in terrains, you may also want to deploy it in a terrain. `flat` means deploying on a flat surface, which should be used when training configuration is `adaptive_en`. Other possible choices are `sslope`, `rslope`, `sup`, `sdown` and `discrete`.
  + --terrain_diff DIFFICULTY: If your terrain choice is not `flat`, then you can specify the difficulty of your terrain, which is the amplitude of the terrain.
  + --model_dir DIR: Specify which trained model to load.
* Deploy a trained policy: The deployment process of a trained model on a Unitree Go1 is the same as the repo [walk these way](https://github.com/Improbable-AI/walk-these-ways).


## Citation <a name="Citation"></a>
If you found this repository useful in your work, consider citing:

```
@article{liang2024adaptive,
  title={Adaptive Energy Regularization for Autonomous Gait Transition and Energy-Efficient Quadruped Locomotion},
  author={Liang, Boyuan and Sun, Lingfeng and Zhu, Xinghao and Zhang, Bike and Xiong, Ziyin and Li, Chenran and Sreenath, Koushil and Tomizuka, Masayoshi},
  journal={arXiv preprint arXiv:2403.20001},
  year={2024}
}
```

<br>

This environment builds on the [walk these way](https://github.com/Improbable-AI/walk-these-ways) by Gaberial Margolis and Pulkit Agrawal, Improbable AI Lab, MIT.