# Code respository ddi_compositionality

To setup the repository the minimal instructions are below.

* Clone the repo: `https://github.com/amansinha09/ddi_compositionality.git`
* Setup the dataset folder
    *  Download the HODDI dataset from `https://github.com/TIML-Group/HODDI.git` 
    * Make sure that files under the following are downloaded because for git big files 
    (`HODDI/dataset/HODDI_v1/dictionary/`: 4 files) and (`HODDI/dataset/HODDI_v1/HODDI/Merged_Dataset`: 2 files)
* Run the `requirement.txt` file: `uv pip install -r requirements.txt`
* Create a file `hftoken` and add your huggingface_token to run hf models.

The above commands are ok to ideally setup things.

To run the default code: `cd script/`
 ```
 python ddi_three_againstall.py
 ```
 This will run the code with model : `Qwen/Qwen2.5-7B-Instruct` over first 500 samples.

# Model covered

|Familty|Models|Order|Remark|
|---|---|---|---|
|Qwen-2.5|7B, 14B, 32B| 2,3,4| 
|llama-3.1|8B|2,3,4|