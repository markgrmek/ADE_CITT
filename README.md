# Classical Integral Transform Technique solver to the multilayer Adevction-Diffusion Equation

```ADE_CITT``` provides the **closed form analytical solution** to the transient (heat) **Advection-Diffusion Equation (ADE)** in a **one-dimensional multi-layer composite medium** of a finite length via the Classical Integral Transform Technique (CITT). 
The solver considers Dirichlet piece-wise linear boundary conditions on the start/top of the domain and either Dirichlet, Neumann or Robin constant boundary conditions at the bottom end of the domain.
This solution was derived for predicting **shallow subsurface thermal field evolution under climate change** and was subsequently applied for predicting groundwater temperatures in Berlin, Germany (sample data provided).

The study is described in detail in [study report](study_report)

## Environment setup

  ```
  python pip install requirements.txt
  ```

## Example code

- Script for validationg the CITT solution is available in [validation.ipynb](validation.ipynb)
- Example script for predicting shallow subsurface thermal field evolution under climate change is availalbe in [example.ipynb](example.py)

# Data Disclaimer
This project uses publicly available data from the following official sources:

Senatsverwaltung für Mobilität, Verkehr, Klimaschutz und Umwelt Berlin
Data obtained from the [Wasserportal Berlin](https://wasserportal.berlin.de/start.php)

Deutscher Wetterdients (DWD)
Data accessed via the [climate data](https://www.dwd.de/EN/ourservices/cdc/cdc_ueberblick-klimadaten_en.html)

The original data remain the property of their respective providers. The versions included here have been cleaned, processed, and reformatted for the purposes of this project. This repository does not claim ownership of the original data. Users seeking authoritative or updated versions of the datasets should consult the respective official data portals.
