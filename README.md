🏁 Background Function Connectivity Analyses Kit 
=======

bgfc_kit is a python package for designed for background functional connectivity analyses

[![Python](https://img.shields.io/badge/Python-3.9+-brightgreen.svg?style=flat-square)](http://shields.io)

*Author*:       [Peetal Li](http://github.com/peetal) <br>
*Code repository*:   [https://github.com/peetal/bgfc_kit](https://github.com/peetal/bgfc_kit) <br>
*Package documentation*: [https://peetal.github.io/bgfc_kit/](https://peetal.github.io/bgfc_kit/) <br>

☀ Introduction
-----------------------------

Function connectivity (FC) is a common neural measure in functional MRI analyses, which measures how different parts of the brain interact with each other. Measuring FC during task can be challenging due to coactivation confounds. That is, spurious interactions can be created between two brain regions because they both react to an external stimuli. One approach to overcome this problem is to compute "background functional connectivity", by first modeling and removing the stimuli-evoked component from the measured timeseries and then computing functional connectivity from the "residual" timeseries (see one of the first bgfc papers [here](https://doi.org/10.1073/pnas.1202095109) and my work [here](https://doi.org/10.1016/j.neuroimage.2023.120221)). 

While this method has been introduced, the exact execution and nuiances have not been clearly defined and documented. _bgfc_kit_ is a python package introduced to fill this gap and to provide all necessary tools for conducting background functional connectivity analyses. Specifically, this package includes three modules, covering tools for design matrix generation, configuring and running a well designed and tested preprocessing pipeline, and multiple functions for turning the preprocessed timeseires into epoch-wise bgfc matrices. The _bgfc_kit_ package aims to standarize and provide a handy tool and easy starting point for future researchers to perform background functional connectivities for their research. 

✺  Installation instructions
-----------------------------
The following is probably the simplest and most direct way to install Nostril on your computer or cluster:
```
pip install git+https://github.com/peetal/bgfc_kit.git
```

📚 More information
-----------------

Please see [documentation](https://peetal.github.io/bgfc_kit/) for detailed information of each module. <br>
Please see [demo](https://github.com/peetal/bgfc_kit/tree/main/bgfc_kit/demo) subdirectory for a detailed demo of key features. <br>
