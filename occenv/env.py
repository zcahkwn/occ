from random import sample

class DataShare:
    def __init__(self, total_number: int):
        self.total_number = total_number
        self.secret_set = range(total_number)  


    def create_shard(self, shard_size: int) -> list[int]:
        # require shard_size <= total_number
        assert shard_size <= self.total_number, "Shard size should be less than total number of data points"
        # randomly sample shard_size data points from the secret set without replacement
        return sample(self.secret_set, shard_size)


    
if __name__ == "__main__":
    new_mpc = DataShare(100)
    alice_shard = new_mpc.create_shard(30)
    bob_shard = new_mpc.create_shard(40)
    charlie_shard = new_mpc.create_shard(50)
    # check whether collusion can be successiful
    set(alice_shard + bob_shard + charlie_shard) == set(new_mpc.secret_set)

    
