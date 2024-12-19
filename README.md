# ChemEnzyRetroPlanner

<div id="top" align="center">

  <h3>ChemEnzRretroPlanner: An Integrated Automation Platform for Hybrid Organic-Enzymatic Synthesis Planning</h3>
  
  [![GitHub Repo stars](https://img.shields.io/github/stars/wangxr0526/ChemEnzyRetroPlanner?style=social)](https://github.com/wangxr0526/ChemEnzyRetroPlanner/stargazers)
  [![WebSite](https://img.shields.io/badge/WebSite-blue)](http://easifa.iddd.group/)
  ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
  <!-- [![DOI](https://zenodo.org/badge/745387829.svg)](https://zenodo.org/doi/10.5281/zenodo.12819439) -->

</div>

![ChemEnzyRetroPlanner](./images/retroplanner-beta-overview.png)



## Contents

- [ChemEnzyRetroPlanner](#chemenzyretroplanner)
  - [Contents](#contents)
  - [Publication](#publication)
  - [Web Server](#web-server)
  - [OS Requirements](#os-requirements)
  - [Python Dependencies](#python-dependencies)
  - [Installation Guide](#installation-guide)
  - [Cite Us](#cite-us)
 

## Publication
ChemEnzRretroPlanner: An Integrated Automation Platform for Hybrid Organic-Enzymatic Synthesis Planning

## Web Server

We have developed a [WebServer](http://easifa.iddd.group) for ChemEnzyRetroPlanner, which allows you Utilize our hybrid synthetic planning platform, which leverages mixed organic enzyme catalysis, to achieve more efficient and environmentally friendly synthesis planning.<br>

<div id="top" align="center">

![ChemEnzyRetroPlanner input interface](webapp/static/figure/input_interface.png)

<h4>ChemEnzyRetroPlanner input interface</h4><br>

![ChemEnzyRetroPlanner results interface](webapp/static/figure/results_interface.png)

<h4>ChemEnzyRetroPlanner results interface</h4><br>

![ChemEnzyRetroPlanner queue interface](webapp/static/figure/queue_interface.png)

<h4>ChemEnzyRetroPlanner queue interface</h4><br>

</div>


## OS Requirements
This repository has been tested on **Linux**  operating systems.

## Python Dependencies
* Python (version >= 3.8)
* DGL (version 2.0.0, CUDA 12.1)
* Torch (version 2.1.2, CUDA 12.1)
* RDKit (version 2022.9.5)
* NumPy (version >= 1.23.5)
* Pandas (version 1.4.4)


## Installation Guide

It is recommended to use conda to manage the virtual environment.The installation method for conda can be found [here](https://conda.io/projects/conda/en/stable/user-guide/install/linux.html#installing-on-linux).<br>

```
git clone https://github.com/wangxr0526/ChemEnzyRetroPlanner.git
cd ChemEnzyRetroPlanner
chmod +x ./setup_ChemEnzyRetroPlanner.sh
./setup_ChemEnzyRetroPlanner.sh

```
Build the Parrot service image
```
cd ChemEnzyRetroPlanner/retro_planner/packages/parrot
chmod +x ./build_parrot_in_docker.sh
./build_parrot_in_docker.sh
```
Start ChemEnzyRetroPlanner services

```
cd ChemEnzyRetroPlanner/docker
chmod +x ./run_container.sh
./run_container.sh
```
The ChemEnzyRetroPlanner server is deployed at http://localhost:8001






```

```
## Cite Us

```

```
