import kagglehub

path = kagglehub.dataset_download(
    "clmentbisaillon/fake-and-real-news-dataset"
)

print("Dataset downloaded to:")
print(path)