
def test0():
    import torch
    x0 = torch.tensor([1, 2, 3])
    x1 = torch.zeros((4, x0.shape[0]))
    x1 = x0.repeat(x1.shape[0], 1)
    print(x0)
    print(x1)


test0()
"""
"""