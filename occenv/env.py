from random import randrange
from occenv.constants import DATA_DIR


class DataShare:
    def __init__(self, total_number):
        self.total_number = total_number
        self.parties_num = randrange(2, 5)

    def parties_number(self):
        return self.parties_num
 
    def parties_size(self):
        self.n_i=[randrange(1,self.total_number) for i in range(self.parties_number())]
        return self.n_i

    def create_shard(self):
        pass

    
if __name__ == "__main__":
    ds = DataShare(len(DATA_DIR/'dummy_data.csv'))
    sizes = ds.parties_size()
    print("Number of parties:", ds.parties_number())
    print("Random sizes for each party:", sizes)


