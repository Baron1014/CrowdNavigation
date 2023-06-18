# Social-STGCNRL
**[`Paper`](https://ieeexplore.ieee.org/xpl/conhome/1802664/all-proceedings) | [`Video`](https://www.youtube.com/watch?v=zeqPHfoYtOI)**

## Abstract
Safety and effectiveness play a vital role in the crowd-aware navigation of robots. Previous research in this area has predominantly relied on an omniscient approach to design the environment, resulting in exceptional obstacle avoidance performance. However, this approach neglects the constraints imposed by the limited field of view in real-world applications. Moreover, constructing a robot with a complete field of view can be excessively costly. In light of these challenges, this paper introduces a novel approach that combines a social spatial-temporal graph convolutional neural network (Social- STGCN) with reinforcement learning (RL) to enhance crowd avoidance navigation, offering a more practical and cost-effective solution. The simulations demonstrate the effectiveness of our proposed method, as it achieves higher success rates and lower collision rates compared to existing approaches. This confirms that our algorithm is an efficient and safer solution for crowd avoidance navigation.

## Model Description
<img src="https://i.imgur.com/TZFGyAF.jpg" width="1000" />

## Citation
If you find the codes or paper useful for your research, please cite our paper:
```bibtex
TBD.
```

## References
Part of the code is based on the following repositories:

[1] C. Chen, Y. Liu, S. Kreiss, and A. Alahi, “Crowd-robot interaction: Crowd-aware robot navigation with attention-based deep reinforcement learning,” in International Conference on Robotics and Automation (ICRA), 2019, pp. 6015–6022.
(Github: https://github.com/vita-epfl/CrowdNav)

[2] Liu, Shuijing, et al. "Decentralized structural-rnn for robot crowd navigation with deep reinforcement learning." 2021 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2021.
(Github: https://github.com/Shuijing725/CrowdNav_DSRNN/)

[3] Mohamed, Abduallah, et al. "Social-stgcnn: A social spatio-temporal graph convolutional neural network for human trajectory prediction." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.
(Github: https://github.com/abduallahmohamed/Social-STGCNN)

[4] Wojke, Nicolai, Alex Bewley, and Dietrich Paulus. "Simple online and realtime tracking with a deep association metric." 2017 IEEE international conference on image processing (ICIP). IEEE, 2017.
(Github: https://github.com/nwojke/deep_sort)
