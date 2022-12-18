from Membershipinfer.Membershipdataset import Membershipdataset

if __name__ == "__main__":
    dataset = Membershipdataset()
    print(sum(dataset.labels))