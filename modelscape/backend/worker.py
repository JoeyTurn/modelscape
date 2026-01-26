import torch
from modelscape.backend.utils import tuple_to_numpy
from modelscape.backend.job import run_job

def worker(device_id, job_queue, result_queue, global_config, bfn_config, iterator_names=None):
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(device_id)
        except Exception as e:
            result_queue.put(("bootstrap_err", repr(e)))
            return

    while True:
        item = job_queue.get()
        if item is None:
            break
        try:
            job_index = 0
            job = item
            if isinstance(item, tuple) and len(item) == 3 and item[0] == "job":
                _, job_index, job = item
            payload = run_job(device_id, job, global_config, bfn_config, iterator_names, job_index=job_index)
            payload = tuple_to_numpy(payload)
            
            result_queue.put(("ok", payload))
        except Exception as e:
            result_queue.put(("err", (job, repr(e))))
