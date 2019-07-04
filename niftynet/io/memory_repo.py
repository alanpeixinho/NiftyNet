"""
Keeps in memory images to be accessed faster
@author Peixinho
TODO: cache if needed
"""

class MemoryRepo:
    _data = None

    @staticmethod
    def register_data(data):
        MemoryRepo._data = data
        assert(len(data.shape) == 5)

    @staticmethod
    def read_sample(mem_path):
        file, sample = [int(x) for x in mem_path.split('/')]
        return MemoryRepo._data[file, sample, ...]
