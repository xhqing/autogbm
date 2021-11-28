import os
import time

start_time = time.time()
while True:
    end_time = time.time()
    spend_time = end_time - start_time
    if spend_time > 30:
        raise TimeoutError("test")
        break
