# IPython モジュールをインポートする場合の注意

通常の python インタプリタで `import IPython` を行うと、それ以前にインポートしたモジュールが、インタプリタ終了時に不適切なタイミングで
開放されてしまう。
この問題に起因して、**インタプリタ終了時**に `quel_ic_config` と `e7awghal` モジュール内の各オブジェクトの `__del__()` メソッド内で エラーが発生することがある。
下に示す例では、`TypeError` が2回発生しているが、それぞれ bisect モジュールと threading モジュールが提供する関数が早期に開放されてしまったことが原因である。

```text
TypeError: 'NoneType' object is not callable
Exception ignored in: <function E7awgMemoryObj.__del__ at 0x7f32a645fd30>
Traceback (most recent call last):
  File "/home/sugita-local/daily-local/20240626/20240626venv/lib/python3.9/site-packages/e7awghal/e7awg_memoryobj.py", line 109, in __del__
  File "/home/sugita-local/daily-local/20240626/20240626venv/lib/python3.9/site-packages/e7awghal/e7awg_memoryobj.py", line 63, in _deallocate_liveobj
TypeError: 'NoneType' object is not callable
Exception ignored in: <function Quel1ConfigSubsystemRoot.__del__ at 0x7f32a5b79af0>
Traceback (most recent call last):
  File "/home/sugita-local/daily-local/20240626/20240626venv/lib/python3.9/site-packages/quel_ic_config/quel1_config_subsystem_common.py", line 216, in __del__
  File "/home/sugita-local/daily-local/20240626/20240626venv/lib/python3.9/site-packages/quel_ic_config/exstickge_coap_client.py", line 714, in terminate
  File "/usr/lib/python3.9/threading.py", line 1056, in join
TypeError: 'NoneType' object is not callable
```

この問題は、実行環境を適切に管理することで解決できる。

- `import IPython` を行うコードの実行には、`ipython` (あるいは、`python -m IPython`) を使用する。

通常の python インタプリタの終了時に、`__del__()`関係でエラーが発生する場合には、コードのどこかで IPython のモジュールを
読み込んでいないか確認していただきたい。
よく見かけるのは、jupyter kernel上で使用した画面表示などの関数を含んだファイルを、通常のpythonスクリプトに転用することで意図せずに発生するケースである。
つまり、スクリプトがインポートしたファイルの中で、`from IPython.display import display` をしているケースだ。
ソースファイルがあるディレクトリで、
```shell
rg IPython
```
などとすれば、簡単に発見できる。

IPython モジュールが不要であれば、import を削除することで回避できる。
そうでなければ、上述のとおり、ipython で実行すれば問題は発生しない。
あるいは、python から実行するスクリプトの冒頭で `import IPython` することでも回避できるが、静的解析との相性が悪いので、使いにくいだろう。
