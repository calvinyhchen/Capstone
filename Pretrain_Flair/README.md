# Pretrain Flair embeddings notes:

## 1. Parameters: (18M parameter size)
`hidden_size=2048`, `sequence_length=250`, `mini_batch_size=100`, `max_epochs=50`

All other parameters, we use default.

## 2. Train:
Remember to train both forward and backward model (change the `is_forward_lm` value) 
The training can only run on one GPU now (2019/03). Training our BioFliar would converge when running for 4x epochs which take about 7-8 days. Not that, the code would not terminate itself, maybe the default setting could not fit our experiment, but in our observation the learning rate was already very low and stagnated for many epochs.


## 3. Data:
The data are stored at `/local/kevinshih/BioFlair/data/PMC_Case_Rep/`, and they were collected in the following way:
	1. Collect all files from folders (in PMC dataset) whose name match the term `CASE_REP`.
	2. Randomly sorted the files and split them into 10 splits
	3. Use 1 for dev, 1 for test, and 8 for train.
	4. The folder structure look like this:
	```wiki
	corpus/
	corpus/train/
	corpus/train/train_split_1
	corpus/train/train_split_2
	corpus/train/...
	corpus/train/train_split_X
	corpus/test.txt
	corpus/valid.txt
	```
