from Membershipinfer.Membershipdataset import Membershipdataset
from Membershipinfer.infermembership import Membershipinfer
if __name__ == "__main__":
    membership = Membershipinfer()
    membership.train()
# if __name__ == "__main__":
#     dataset = Membershipdataset()
#     print(sum(dataset.labels))