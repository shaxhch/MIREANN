#作者：Yuwei
#链接：https://www.zhihu.com/question/307282137/answer/1560137140
#来源：知乎
from threading import Thread
import torch
from queue import Queue

class CudaDataLoader:

    def __init__(self, loader, device, queue_size=0):
        self.device = device
        self.loader = loader
        self.idx = 0
        self.load_stream = torch.cuda.Stream(device=device)
        self.queue = Queue(maxsize=queue_size)
        self.worker = Thread(target=self.load_loop)
        self.worker.setDaemon(True)
        self.worker.start()
        self.length=self.loader.length

    def load_loop(self):
        # The loop that will load into the queue in the background
        while True:
            with torch.cuda.stream(self.load_stream):
                for sample in self.loader:
                    self.queue.put(self.load_instance(sample))

    def load_instance(self, sample):
        if torch.is_tensor(sample):
            return sample.to(self.device, non_blocking=True)
        else:
            return (self.load_instance(s) for s in sample)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if not self.worker.is_alive() and self.queue.empty():
            self.idx = 0
            self.queue.join()
            self.worker.join()
            raise StopIteration
        elif self.idx >= self.length:
            self.idx = 0
            raise StopIteration
        else:
            out = self.queue.get()
            self.queue.task_done()
            self.idx += 1
        return out

