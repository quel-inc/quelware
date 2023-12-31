# A Python Library for controlling QuEL Clock Master 

クロックマスタと各制御装置との同期を取るためのライブラリと、そのライブラリをシェルからお手軽に使うためのコマンド群を提供します。

# Build and Install with `pip`

適切なPythonの仮想環境にて、次のコマンドを実行してください。パッケージのビルド及びインストールが完了します。
```shell
cd qubemaster/software
rm -rf dist
python setup.py bdist_wheel
pip install dist/*.whl
```

なお、`pipenv` をお使いの場合には、dist 以下に生成された quel_clock_master*.whl を適切な場所にコピーしてご使用ください。

# Console Commands
パッケージをインストールすると、次の4つのコマンドがシェルから利用できるようになります。
各コマンドを `-h` を付けて実行すると、引数の詳細が表示されます。

- `quel_clock_master_read`: マスタのIPアドレスと複数の各制御装置のIPアドレスを指定して使います。マスタと指定の各制御装置のクロックカウンタを取得します。最初のIPアドレスがマスタ、それ以降のIPアドレスが制御装置であるとみなされます。
- `quel_clock_master_kick`: マスタのIPアドレスと複数の制御装置のIPアドレスを指定して使います。マスタと指定の各制御装置とを同期します。
- `quel_clock_master_clear`: 指定したIPアドレスのマスターのクロックカウンタをゼロにクリアします。複数の制御装置のIPアドレスを同時に引数に与えると、クリアに引き続いて、マスタと各制御装置とを同期します。
- `quel_clock_master_reset`: 指定したIPアドレスのマスターをリセットします。**`kick`などのコマンドが動作しなくなった場合に使うと、マスタが復旧します。** ただし、内部のクロックカウントが0にリセットされるので、マスタと各制御装置との再同期が必要になります。   

## 注意点
`quel_clock_master_kick` に与える制御装置のアドレスに不正なものがあると、クロックマスタ内のステートマシンが一部ハングアップします。
これが発生してしまうと、`quel_clock_master_reset` を実行するまで、`kick`が動作しなくなるので注意が必要です。

## 実行例
### クロックカウンタの読み出し
クロックマスター(10.3.0.13) と2台の制御装置(10.2.0.42, 10.2.0.58) とのそれぞれの内部クロックカウンタの値を表示します。
これらは数時間前に予め同期をしてあります。
```shell
$ quel_clock_master_read 10.3.0.13 10.2.0.42 10.2.0.58
2023-06-26 01:34:43,741 [INFO] quel_clock_master_consoleapps.apps: 10.3.0.13: 179949078638
2023-06-26 01:34:43,741 [INFO] quel_clock_master_consoleapps.apps: 10.2.0.42: 179952338538
2023-06-26 01:34:43,741 [INFO] quel_clock_master_consoleapps.apps: 10.2.0.58: 179952358039
```

#### 解説と余談
クロックカウンタは125MHzで動作しているので、2つの制御装置間のカウンタの差は 0.16ms程度です。
この差のもっとも大きな成分は、各制御装置に個別に問い合わせパケットを送っていることによる時間差です。
一方で、マスターと各制御装置の差は、26ms 程度と桁違いに大きいです。これは、各制御装置が共通のクロックによって駆動されているのに対し、
マスタだけ別のクロックで駆動されていることによります。
さきほど述べたように、同期を行ったのが数時間前なので、クロック周波数の誤差が数時間分蓄積して、カウンタの差として顕在化しています。

マスタは同期の時点で、制御装置に時刻を提供できさえすればよいので、この誤差蓄積は大きな問題にはならないのが普通です。
無理に問題になるケースを挙げるとすれば、既に同期が取れている制御装置のグループに新たな制御装置を同期させる場合があります。
マスタも共通のクロックで駆動されていれば、新たにグループに追加する制御装置だけ同期を取れば済むのが、現状では、 全ての制御装置をマスターに再同期
させるしかありません。マスタだけクロックが別であることは、実用上大きい問題がないとしても、直感的な理解を妨げるので、近い将来に改善する予定です。

### 同期
以下のコマンドで、クロックマスター(10.3.0.13) とさきほどの2台の制御装置(10.2.0.42, 10.2.0.58) を再同期します。
```shell
$ quel_clock_master_clear 10.3.0.13 10.2.0.42 10.2.0.58
2023-06-26 01:46:45,373 [INFO] quel_clock_master_consoleapps.apps: cleared successfully
2023-06-26 01:46:45,373 [INFO] quel_clock_master_consoleapps.apps: kicked successfully
$
$ quel_clock_master_read 10.3.0.13 10.2.0.42 10.2.0.58
2023-06-26 01:46:46,737 [INFO] quel_clock_master_consoleapps.apps: 10.3.0.13: 170499729
2023-06-26 01:46:46,737 [INFO] quel_clock_master_consoleapps.apps: 10.2.0.42: 170524873
2023-06-26 01:46:46,737 [INFO] quel_clock_master_consoleapps.apps: 10.2.0.58: 170549663
```
`quel_clock_master_clear` コマンドで同期を取った直後に、マスタと各制御装置のカウンタを取得すると、全てのカウンタ値が0.2ms程度の誤差で
一致していることが分かります。この誤差は、上で述べたように、時刻問い合わせのタイミングのずれによります。なお、clear コマンドはマスタのクロック
カウントをゼロにしてしまいます。これを避けたい場合には、`quel_clock_master_kick` コマンドを使います。

# APIs
当然ですが、ライブラリをインポートして、実験用のPythonスクリプトから各APIを叩くこともできます。
APIの詳細については、文書化できていないので、ソースコードをご覧ください。
[コンソールコマンドのコード](quel_clock_master_consoleapps/apps.py) と、上記の説明を合わせると理解しやすいと思います。
また、ライブラリ中のクラスを定義しているファイルをインタプリタで直接実行すると、ログレベルがDEBUGになった状態で、いろいろテスト実行を
行なえます。動的な振る舞いを理解するのに便利です。
仕様の細かい部分については、tests 及び tests_with_devices 以下のテストケースが参考になります。
