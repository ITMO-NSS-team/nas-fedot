# nas-fedot

---

![License](https://img.shields.io/github/license/ITMO-NSS-team/nas-fedot?style=flat&logo=opensourceinitiative&logoColor=white&color=blue)
[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

Built with:

![keras](https://img.shields.io/badge/Keras-D00000.svg?style={0}&logo=Keras&logoColor=white)
![numpy](https://img.shields.io/badge/NumPy-013243.svg?style={0}&logo=NumPy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458.svg?style={0}&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikitlearn-F7931E.svg?style={0}&logo=scikit-learn&logoColor=white)

---

## Overview

nas-fedot automates the design of neural network architectures, specifically targeting improved performance in challenging remote sensing applications like satellite image analysis. Built upon the Fedot AutoML framework, this work introduces FEDOT-NAS, an evolutionary approach to Neural Architecture Search (NAS) that intelligently creates convolutional neural networks tailored for object recognition. By automating architecture engineering, it aims to surpass the capabilities of manually designed or conventionally searched models such as InceptionV3 and ResNet152V2, and even other NAS frameworks like AutoKeras and NNablaNas. The openly available implementation facilitates further research into automated machine learning solutions for satellite imagery and beyond.

---

## Table of Contents

- [Content](#content)
- [Algorithms](#algorithms)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [License](#license)
- [Citation](#citation)

---
## Content

The nas-fedot project focuses on automating the design of neural network architectures, specifically for tasks like object recognition in satellite imagery. It leverages an evolutionary algorithm to search for optimal configurations represented as convolutional and classification layers. The system integrates with the broader Fedot AutoML framework, utilizing its caching mechanisms for efficiency. Key components include a composer that orchestrates the search process, dataset builders for data preparation, and a model constructor to build models from optimized architectures. Evaluation is performed using datasets like Statoil and BigEarth, comparing performance against established models such as InceptionV3 and ResNet152V2, alongside other NAS frameworks.

---

## Algorithms

This project implements a Neural Architecture Search (NAS) methodology, specifically an evolutionary approach for automatically designing convolutional neural networks. The core technique utilizes a genetic algorithm to explore diverse network architectures, representing CNNs as graphs and evolving them through iterative optimization. An objective function evaluates the performance of each generated architecture on satellite imagery datasets. A key component is the use of pipelines and caching mechanisms to improve efficiency during model evaluation. Finally, the system employs multi-objective optimization to refine the search process and enhance overall performance in remote sensing tasks.

---

## Installation

Install nas-fedot using one of the following methods:

**Build from source:**

1. Clone the nas-fedot repository:
```sh
git clone https://github.com/ITMO-NSS-team/nas-fedot
```

2. Navigate to the project directory:
```sh
cd nas-fedot
```

3. Install the project dependencies:

```sh
pip install -r requirements.txt
```
## Getting Started

The implementation of Neural Architecture Search based at the Fedot framework. You can find the Fedot source code here: https://github.com/nccr-itmo/FEDOT.

A video tutorial is available here: https://www.youtube.com/watch?v=CkzzZbIvwH0 (in Russian).

---

## License

This project is protected under the BSD 3-Clause "New" or "Revised" License. For more details, refer to the [LICENSE](https://github.com/ITMO-NSS-team/nas-fedot/tree/master/LICENSE) file.

---

## Citation

If you use this software, please cite it as below.

### APA format:

    ITMO-NSS-team (2020). nas-fedot repository [Computer software]. https://github.com/ITMO-NSS-team/nas-fedot

### BibTeX format:

    @misc{nas-fedot,

        author = {ITMO-NSS-team},

        title = {nas-fedot repository},

        year = {2020},

        publisher = {github.com},

        journal = {github.com repository},

        howpublished = {\url{https://github.com/ITMO-NSS-team/nas-fedot.git}},

        url = {https://github.com/ITMO-NSS-team/nas-fedot.git}

    }

---
