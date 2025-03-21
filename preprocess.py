# import threading
# import time
# import random

# def cpu_stress():
#     while True:
#         [random.random() ** 2 for _ in range(10**6)]  

# num_threads = 16 

# threads = []
# for _ in range(num_threads):
#     t = threading.Thread(target=cpu_stress)
#     t.daemon = True  
#     t.start()
#     threads.append(t)

# try:
#     while True:
#         time.sleep(1)
# except KeyboardInterrupt:
#     print("preprocessing complete")
