import torch

def printMemoryUsed(abbr=False):
    print(torch.cuda.memory_summary(device=torch.cuda.current_device(), abbreviated=True))

if __name__ == "__main__":
    printMemoryUsed(True)
    print(torch.cuda.current_device())


