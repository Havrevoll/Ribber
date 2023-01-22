# import multiprocessing as mp
# import pickle

# from datagenerering import tre_objekt

# def function(inputs, shared_object):
#     # Access and modify the shared object
#     shared_object.points[0, 0] += 1
#     result = shared_object.points[0, 0] + inputs
#     return result

# # Load the pickled object
# with open('data/rib75_Q20_1.pickle', 'rb') as f:
#     shared_object = pickle.load(f)

# # Convert the object to a shared object
# manager = mp.Manager()
# shared_object = manager.Value(tre_objekt, shared_object)

# # Define the input data
# input_data = [1, 2, 3, 4, 5]

# # Create a pool of processes
# with mp.Pool(4) as p:
#     # Apply the function to the input data in parallel
#     results = p.starmap(function, [(inputs, shared_object) for inputs in input_data])

# print(results)


from datagenerering import tre_objekt
import pickle
import numpy as np
import multiprocessing as mp

with open('data/rib75_Q20_1.pickle', 'rb') as f:
    shared_object = pickle.load(f)

def funk(inn, shared_object):
    t,u = shared_object.get_tri_og_U(inn)
    print(inn, t.find_simplex([inn,0,0]))
    return inn + int(t.find_simplex([inn,0,0,]))

# ta,ua = shared_object.get_tri_og_U(0)
print(funk(1,shared_object))

print("hadet")