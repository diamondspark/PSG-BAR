# PSG-BAR

![PSG-BAR Architecture diagram] (https://github.com/diamondspark/PSG-BAR/blob/main/architecture.png?raw=true)

PSG-BAR: Protein Structure Graph- Binding Affinity Regression is geometric deep learning inspired tool utilizing 3D structure of proteins to predict binding affinity of protein-ligand complexes.
https://doi.org/10.1101/2022.04.27.489750
## Requirements
Cuda enabled GPU. Recommended Tesla V100 (it's equivalent or above).
## Installation 
```
>> bash setup.sh
>> conda activate psg-bar
```
## Training Model
### For Binding Affinity regression on custom dataset
Create a dataset similar to [here](/data/DPI/PDBBind_Sample/sample_pdbbind_dataset.csv). Then follow instructions in [Demo-Elements](Demo-Elements.ipynb) to train the model.

## Citation
```
@article {Pandey2022.04.27.489750,
	author = {Pandey, Mohit and Radaeva, Mariia and Mslati, Hazem and Garland, Olivia and Fernandez, Michael and Ester, Martin and Cherkasov, Artem},
	title = {Ligand Binding Prediction using Protein Structure Graphs and Residual Graph Attention Networks},
	elocation-id = {2022.04.27.489750},
	year = {2022},
	doi = {10.1101/2022.04.27.489750},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Motivation: Computational prediction of ligand-target interactions is a crucial part of modern drug discovery as it helps to bypass high costs and labor demands of in vitro and in vivo screening. As the wealth of bioactivity data accumulates, it provides opportunities for the development of deep learning (DL) models with increasing predictive powers. Conventionally, such models were either limited to the use of very simplified representations of proteins or ineffective voxelization of their 3D structures. Herein, we present the development of the PSG-BAR (Protein Structure Graph Binding Affinity Regression) approach that utilizes 3D structural information of the proteins along with 2D graph representations of ligands. The method also introduces attention scores to selectively weight protein regions that are most important for ligand binding. Results: The developed approach demonstrates the state-of-the-art performance on several binding affinity benchmarking datasets. The attention-based pooling of protein graphs enables identification of surface residues as critical residues for protein-ligand binding. Finally, we validate our model predictions against an experimental assay on a viral main protease (Mpro) the hallmark target of SARS-CoV-2 coronavirus. Availability: The code for PSG-BAR is made available at https://github.com/diamondspark/PSG-BARCompeting Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2022/04/28/2022.04.27.489750},
	eprint = {https://www.biorxiv.org/content/early/2022/04/28/2022.04.27.489750.full.pdf},
	journal = {bioRxiv}
}
```
