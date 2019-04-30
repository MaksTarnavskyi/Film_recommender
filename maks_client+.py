
# coding: utf-8

# In[1]:


import numpy as np
import socket
import sys
import time
from tabulate import tabulate


# In[2]:


def send(text):
    sock.sendall(text.encode("utf-8"))
    
def receive():
    data = sock.recv(1024)
    udata = data.decode("utf-8")
    return udata


# # Client

# In[ ]:


sock = socket.socket()
sock.connect(('localhost', 9091))
print(receive())
send('Start')
ans = receive()
while ans != 'bye':
    if ans == 'pr':
        send('OK')
        ans = receive()
        print(ans)
        send('OK')
    if ans == 'in':
        send('OK')
        ans = receive()
        send('wait')
        time.sleep(0.5)
        res = input(ans)
        send(res)
    if ans == 'tab':
        send('OK')
        res = ''
        ans = receive()
        while ans != 'end':
            res += ans
            send('OK')
            ans = receive()
        send('wait')
        print(res)
        send('OK')
        
    ans = receive()

sock.close()

