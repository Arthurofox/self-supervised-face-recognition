from datasets import load_dataset

# Load the dataset from Hugging Face (this returns a DatasetDict with splits, e.g., train, test)
ds = load_dataset("FastJobs/Visual_Emotional_Analysis")
print(ds)
