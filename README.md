HIM: Discovering Implicit Relationships in Heterogeneous Social Networks
========
This repository contains an implementation of the Implicit Relation Discovery Framework (HIM) for heterogeneous networks, as well as experiments on Implicit Relation Prediction. A description of the model and results can be found in the paper:

Requirements
--------
HIM relies on Python 3.6, PyTorch 1.7.0 and Tensorflow 1.8.0. For detailed requirements, please refer to requirements.txt.

Datasets
--------
* Same-City Relationship
* Advisor-Advisee Relationship
* Terrorist Attacks

Same-City Relationship and Advisor-Advisee Relationship can be downloaded from the following link：
[HIMdata](https://github.com/myjpgit/HIMdata.git)

Data preprocessing
---------

### Same-City Relationship dataset (same_city folder):
The metadata includes:
* refined_data/data.txt
* refined_data/node_features.txt
* refined_data/node_ind.txt
* data/relation_true.txt
* data/relation_false.txt

Run python generate_data.py to generate refined_data/true_false.txt, refined_data/train_data.txt, refined_data/test_data.txt and data/train_true_data.txt

Run python generate_vector.py to generate refined_data/node_features.mat and refined_data/one_hot_vector.mat

Run python refine_adj.py to generate refined_data/node_node.mat

Run python generate_edge_feature.py to generate refined_data/adj.npy and refined_data/efeature.npy

Run python generate_laplacian.py to generate refined_data/node_laplacian.mat

Run python generate_aa_list.py to generate ANSWER/HetGNN_samecity/data/academic/a_a_list_train.txt

Run python generate_neigh.py to generate ANSWER/HetGNN_samecity/data/academic/het_neigh.txt

### Same-City Relationship dataset (HetGNN_samecity folder):
Before processing, please copy the ANSWER/same_city/refined_data/node_features.mat to the ANSWER/HetGNN_samecity/data/academic/node_features.mat.

Run python het_random_walk_generate.py to generate data/academic/het_random_walk.txt

Run python HetGNN.py to generate data/academic/node_embeddings.txt

Run python generate_end_embed.py to generate ANSWER/same_city/refined_data/end_node_embeddings.mat

The above is the preprocessing method of the Same-City Relationship dataset

-----------------------------------------------------------------------------------------------------------------------------------

### Advisor-Advisee Relationship dataset (advisor_advisee folder):
The metadata includes:
* refined_data/data.txt
* refined_data/aut_institution.mat
* refined_data/author_paper.mat
* refined_data/paper_attribute.mat
* refined_data/node_ind.txt
* refined_data/author_index.txt
* refined_data/one_hot_vector.mat
* data/relation_true.txt
* data/relation_false.txt

Run python generate_data.py to generate refined_data/advisor_collaborator.txt, refined_data/train_data.txt, refined_data/test_data.txt and data/train_true_data.txt

Run python refine_adj.py to generate refined_data/author_author.mat

Run python generate_edge_feature.py to generate refined_data/adj.npy and refined_data/efeature.npy

Run python generate_laplacian.py to generate refined_data/node_laplacian.mat

Run python generate_aa_list.py to generate ANSWER/HetGNN_advisor/data/academic/a_a_list_train.txt

Run python generate_neigh.py to generate ANSWER/HetGNN_advisor/data/academic/het_neigh.txt

### Advisor-Advisee Relationship dataset (HetGNN_advisor folder):
Before processing, please copy the ANSWER/advisor_advisee/refined_data/author_paper.mat to the ANSWER/HetGNN_advisor/data/academic/author_paper.mat, copy the ANSWER/advisor_advisee/refined_data/aut_institution.mat to the ANSWER/HetGNN_advisor/data/academic/aut_institution.mat.

Run python het_random_walk_generate.py to generate data/academic/het_random_walk.txt

Run python HetGNN.py to generate data/academic/aut_institutions.txt

Run python HetGNN.py to generate data/academic/author_papers.txt (Need to change the path on line 244 in the tools.py and on line 41 in the data_generator.py)

Run python generate_end_institution.py to generate ANSWER/advisor_advisee/refined_data/end_aut_institution.mat

Run python generate_end_paper.py to generate ANSWER/advisor_advisee/refined_data/end_author_paper.mat

The above is the preprocessing method of the Advisor-Advisee Relationship dataset

-----------------------------------------------------------------------------------------------------------------------------------

### Terrorist Attacks dataset (terror_attack folder): 
The metadata includes:
* refined_data/data.txt
* refined_data/node_feature.mat
* refined_data/node_ind.txt
* data/relation_true.txt
* data/relation_false.txt

Run python generate_data.py to generate refined_data/true_false.txt, refined_data/train_data.txt, refined_data/test_data.txt and data/train_true_data.txt

Run python generate_vector.py to generate refined_data/one_hot_vector.mat

Run python refine_adj.py to generate refined_data/node_node.mat

Run python generate_edge_feature.py to generate refined_data/adj.npy and refined_data/efeature.npy

Run python generate_laplacian.py to generate refined_data/node_laplacian.mat

Run python generate_aa_list.py to generate ANSWER/HetGNN_terror/data/academic/a_a_list_train.txt

Run python generate_neigh.py to generate ANSWER/HetGNN_terror/data/academic/het_neigh.txt

### Terrorist Attacks dataset (HetGNN_terror folder):
Before processing, please copy the ANSWER/terror_attack/refined_data/node_feature.mat to the ANSWER/HetGNN_terror/data/academic/node_feature.mat.

Run python het_random_walk_generate.py to generate data/academic/het_random_walk.txt

Run python HetGNN.py to generate data/academic/node_embedding.txt

Run python generate_end_embed.py to generate ANSWER/terror_attack/refined_data/end_node_embedding.mat

The above is the preprocessing method of the Terrorist Attacks dataset

-----------------------------------------------------------------------------------------------------------------------------------

Model training
-----------

#### ANSWER for Same-City Relationship dataset
Run ANSWER/GCN_ef/main_samecity.py

#### ANSWER for Advisor-Advisee Relationship dataset
Run ANSWER/GCN_ef/main_aa.py

#### ANSWER for Terrorist Attacks dataset
Run ANSWER/GCN_ef/main_ta.py

-----------------------------------------------------------------------------------------------------------------------------------

#### ANSWER-LAP for Same-City Relationship dataset
Run ANSWER/GCN_LAP/main_samecity.py

#### ANSWER-LAP for Advisor-Advisee Relationship dataset
Run ANSWER/GCN_LAP/main_aa.py

#### ANSWER-LAP for Terrorist Attacks dataset
Run ANSWER/GCN_LAP/main_ta.py

-----------------------------------------------------------------------------------------------------------------------------------

#### ANSWER-WOHET for Same-City Relationship dataset
Run ANSWER/GCN_WOHET/main_samecity.py

#### ANSWER-WOHET for Advisor-Advisee Relationship dataset
Run ANSWER/GCN_WOHET/main_aa.py

#### ANSWER-WOHET for Terrorist Attacks dataset
Run ANSWER/GCN_WOHET/main_ta.py

-----------------------------------------------------------------------------------------------------------------------------------

#### ANSWER-WOLAW for Same-City Relationship dataset
Run ANSWER/GCN_WOLAW/main_samecity.py

#### ANSWER-WOLAW for Advisor-Advisee Relationship dataset
Run ANSWER/GCN_WOLAW/main_aa.py

#### ANSWER-WOLAW for Terrorist Attacks dataset
Run ANSWER/GCN_WOLAW/main_ta.py
