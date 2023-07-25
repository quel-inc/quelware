import json
import urllib.error
import urllib.request
from base64 import b64decode

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from quel_inst_tool import E4405bReadableParams, E4405bWritableParams

server_host = "192.168.235.5:8000"

try:
    with urllib.request.urlopen(f"http://{server_host}/param") as response:
        body1 = json.loads(response.read())
        headers1 = response.getheaders()
        status1 = response.getcode()
        print(headers1)
        print(status1)
        print(body1)
except urllib.error.URLError as e:
    print(e.reason)

param = E4405bReadableParams(**body1)


update = E4405bWritableParams(sweep_points=801)
headers = {
    "Content-Type": "application/json",
}
req = urllib.request.Request(f"http://{server_host}/param", update.json().encode(), headers)
try:
    with urllib.request.urlopen(req) as response:
        body2 = json.loads(response.read())
        headers2 = response.getheaders()
        status2 = response.getcode()
        print(headers2)
        print(status2)
        print(body2)
except urllib.error.URLError as e:
    print(e.reason)


try:
    with urllib.request.urlopen(f"http://{server_host}/trace") as response:
        body3 = json.loads(response.read())
        headers3 = response.getheaders()
        status3 = response.getcode()
        print(headers3)
        print(status3)
        print(body3)
except urllib.error.URLError as e:
    print(e.reason)


fd0 = np.frombuffer(b64decode(body3["trace"]))
fd0 = fd0.reshape((fd0.shape[0] // 2, 2))
mpl.use("Qt5Agg")
plt.cla()
plt.plot(fd0[:, 0], fd0[:, 1])
plt.show()
