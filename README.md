# Enhanced dispersion of active microswimmers in confined flows

## Codes 

Hi ! The codes in this repository were used for experimental and numerical data treatment as well as numerical data simulation. Below is an explanation of how to use the codes.

Please begin by downloading all code to a same folder. Code was written for python 3.13.5. 

I recommend working in a venv for convenience, and personally manage all my scripts using VScode on OS. You will also need to ```pip install numpy scipy cython seaborn scikit-learn pandas``` if I recall correctly.

- For the numerical simulations :
  Run ```setup.py``` in your terminal to compile ```chlamy_packages.pyx``` file. Info is in the script. 
  No need to touch ```chlamy_run_simul.py```. It's fine as is and being called by another file.
  Tu run the simulations : simply run ```chlamy_runner_classes.ipynb```. It's a jupyter file. You can tune all the parameters you wanna iterate through easily in it.
  Then, analysis is done with the ```chlamy_analysis_classes.ipynb```. It is not a very elegant code, but it is sufficient and quick enough that I didn't find the need to make it better.

- For Marc's experimental codes :
  His data is available upon reasonable request. Positions of tracked particles are retrieved with ```just_tracking_big_files.py``` and linked into trajectories with ```just_linking_big_files.py```. Then, article figures are created with ```article.py```.


If you find yourself doubting anything, you want precisions or data, please let us know. Marc Lagoin (marc.lagoin@u-bordeaux.fr), Yacine Amarouchene (yacine.amarouchene@u-bordeaux.fr), Antoine Allard (antoine.allard@u-bordeaux.fr) and Thomas Salez (thomas.salez@u-bordeaux.fr) are corresponding authors, but I can also help on code inquiries as the team numerician happily.

Taylor-Aris-ly yours,

Juliette Lacherez (juliette.lacherez@u-bordeaux.fr), with Marc Lagoin, Guirec de Tournemire, Ahmad Badr, Yacine Amarouchene, Antoine Allard and Thomas Salez.

