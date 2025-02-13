import json
from random import randrange, sample
from occenv.constants import DATA_DIR


with open (DATA_DIR/'dummy_20.json', 'r') as f:
    data=json.load(f)

class DataShare:
    def __init__(self, total_number):
        self.total_number = total_number
        self.parties_num = 3

    def parties_number(self):
        return self.parties_num
 
    def parties_size(self):
        self.n_i=[randrange(1,self.total_number) for i in range(self.parties_number())]
        return self.n_i

    def create_shard(self,data):
        shards = []
        for size in self.n_i:
            shard = sample(data, k=size)
            shards.append(shard)
        return shards

    
if __name__ == "__main__":
    ds = DataShare(len(data))
    sizes = ds.parties_size()
    print("Total number of data points:", ds.total_number)
    print("Number of parties:", ds.parties_number())
    print("Random sizes for each party:", sizes)
    shards = ds.create_shard(data)
    for i, shard in enumerate(shards, start=1):
        print(f"Shard {i} (size {len(shard)}):", shard)

