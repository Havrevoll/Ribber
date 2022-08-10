# import sys
from pathlib import Path
import pickle
from multiprocessing.managers import SyncManager
# import numpy as np

from datagenerering import tre_objekt

# Etter mal frå https://stackoverflow.com/questions/47837206/sharing-a-complex-python-object-in-memory-between-separate-processes

tre = tre_objekt

# Proxy class to be shared with different processes
class tre_proxy(object):
    def __init__(self) -> None:
        pass

    def get_tri_og_U(self, t):
        global tre
        # print(t)
        return tre.get_tri_og_U(t)

    def get_ribs(self):
        global tre
        # print("hentar ribber")
        return tre.ribs

    def get_kd_U(self, tx):
        global tre
        # print(tx)
        return tre.get_kd_U(tx)

# class to encapsulate the server functionality
class tre_server(object):
    def __init__(self, port, fname) -> None:
        self.port = port
        self.load(fname)

    # Load the tree into tre (global)
    @staticmethod
    def load(pickle_fil):
        global tre
        with open(pickle_fil,'rb') as f:
            tre = pickle.load(f)

    # Run the server
    def run(self):
        class my_manager(SyncManager): pass
        my_manager.register('tre_proxy', tre_proxy)
        mgr = my_manager(address=('', self.port),authkey=b'tre_proxy01')
        server = mgr.get_server()
        server.serve_forever()

if __name__ == '__main__':
    port = 5010
    fname =  Path("./data/rib50_Q40_1_scale40.pickle")

    print("Hentar tre")
    ts = tre_server(port,fname)

    print("Leverer data. Trykk ctrl-c for å stoppa.")
    ts.run()
