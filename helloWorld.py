#hello
import heapq

def test(a):
    a = [1,1,1]
    return a


values_action = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
a = heapq.nlargest(len(values_action), range(len(values_action)), values_action.__getitem__)
print(a)


print("Hello!")