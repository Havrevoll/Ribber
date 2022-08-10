import sys
from pathlib import Path
import pickle
from multiprocessing.managers import BaseManager
import psutil


# Etter mal frå https://stackoverflow.com/questions/47837206/sharing-a-complex-python-object-in-memory-between-separate-processes

# Grab the shared proxy class.  All methods in that class will be availble here
class tre_client(object):
    def __init__(self, port) -> None:
        assert self._check_for_process('tre_server.py'), "tre_server må køyra fyrst"
        class my_manager(BaseManager): pass
        my_manager.register('tre_proxy')
        self.mgr = my_manager(address=('localhost', port), authkey=b'tre_proxy01')
        self.mgr.connect()
        self.tre = self.mgr.tre_proxy()

    # Return the instance of the proxy class
    @staticmethod
    def get_tre_client(port):
        return tre_client(port).tre

    # Verify the server is running
    @staticmethod
    def _check_for_process(name):
        for proc in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
            if proc.name() == 'python3' and proc.info['cmdline'][1] == name:
                return True
        return False


if __name__ == '__main__':
    port = 5010
    tre = tre_client.get_tre_client(port)

    tri, u = tre.get_tri_og_U(0)
    print(tri.min_bound)


# t_span = (0,179)

# pickle_fil = Path("./data/rib50_Q40_1_scale40.pickle")
# hdf5_fil = pickle_fil.with_suffix(".hdf5")


# shm_a = shared_memory.SharedMemory(create=True, size=6500000000)

# buffer = shm_a.buf
# print(len(buffer))
