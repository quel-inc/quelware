# jtagterminal for Python
## 概要
本ライブラリ(quel_pyxsdb)は、Xilinx の FPGA上のCPUのコンソールを microblaze debug moduleを介して
Pythonスクリプトから叩くための支援を提供する。 具体的には、次のような機能を持つ。
- 指定のIDを持つJTAGアダプタのコンソールをソケットと接続して、そのポート番号を取得する。
    - つまり、telnetlib などでコンソールを握るための準備をする。 
- 複数バージョンのVivadoのhw_serverの共存を容易に行うためのコマンド群を提供する。
    - QuEL社内で頻繁に用いるアダプタの種類ごとに、異なるバージョンのhw_server を立ち上げることが容易にできる。
    - 付加的機能として、特定のIDを持つAlveo U50だけを握ったhw_serverを立ち上げることもできるので、ファームウェアの並列書き込みのお膳立ても支援する。
    
## 想定シナリオ
### シナリオ１
Vivado 2019.1 で開発した CMod A7 用のファームウェアのターミナルを介して、FPGA上のCPUをホストから自動制御して実験をしたい。
実験は半日くらい続くのだが、その途中に、Vivado 2020.1 で開発した Alveo U50用のファームウェアの書き込みをしたくなってしまった。

#### はまりどころ
CModのコンソールを握るためには2019.1用のhw_serverが必要になる。
ここで、2020.1のhw_serverを使うと、とりあえず使えるのだが、時間が立つと謎の不具合が発生して実験が止まってしまう。
一方で、Vivado 2020.1 用に書いた tcl スクリプトで実験に使っているのとは別のFPGAへファームウェアを書き込もうとすると、
2019.1 の hw_server はバージョン不整合で使用できない。
そこで、2020.1 の hw_server を立ち上げたくなるのだが、 単純に立ち上げると競合が発生して、実験スクリプトが止まったり、書き込みに失敗したり、
なんにもうまくいかない。

#### なんでか？
一般的によく行われる次の手順について概観する。
```bash
source /tools/Vivado/2019.1/settings64.sh
xsct
```
としたあとで、xsctのコンソール内で、`conn` すると、2019.1 の hw_server が自動的に起動する。
この hw_server はデフォルトのポート3121を開け、__ホスト上の全てのJtagアダプタを握る__。

この状態で、vivado 2020.1 を使って書き込みを使用とすると、vivadoは、この2019.1 の hw_server
へ接続を試みて失敗する。
かといって、2020.1 の hw_server を立ち上げても、全てのJTAGアダプタが既に握られているので、
リソースの競合が起こり、両方のhw_serverが機能不全を起こす！

#### 通常の解決法
2019.1 の hw_server の起動時に、次のような引数を与えることで、Cmod A7以外を握らないようにする。
```
hw_server -e "set jtag-port-filter Digilent/Cmod A7 - 35T" -s tcp::3121
``` 
そうした上で、2020.1 の hw_server は Alveo U50だけを握るように、同様の引数を与える。

頑張ればできるのだが、面倒くさいし、間違いやすい。
そして、間違えると半日かかる実験が中断する可能性がある！

### シナリオ２
複数のAlveo U50 にファームウェアを書き込みたいが、ひとつあたり30分程度の書き込み時間が
かかるのをなんとかしたい。

#### はまりどころ
書き込みに際して hw_server が必要になるが、何も考えずにやると自動的に起動する hw_server が全てのJTAGアダプタを握って
しまい、並列書き込みができない。

#### 通常の解決法
これも、特定のJTAG IDのAlveo U50以外を握らない複数のhw_serverを立ち上げればよい。
```bash
source /tools/Xilinx/Vivado/2020.1/settings64.sh
hw_server -e "set jtag-port-filter Xilinx/Alveo-DMBv1 FT4232H/50xxxxxAA" -s tcp::4121
hw_server -e "set jtag-port-filter Xilinx/Alveo-DMBv1 FT4232H/50yyyyyAA" -s tcp::5121
```
などとして、書き込みスクリプトを接続先のポートを変えて複数起動すればよい。

## ライブラリの使い方
### 環境構築
Pythonの仮想環境を作成して、pyxsctのパッケージをインストールする。

```bash
cd your_working_directory

python3.9 -m venv your_venv
source your_venv
pip install -U pip
pip install build

git clone git@github.com:quel-inc/sugita_experimental.git
cd sugita_experimental/quel_pyxsdb

python -m build
pip install dist/quel_pyxsdb-0.2.0-py3-none-any.whl
```

この仮想環境では、pyxsct パッケージが使えるのに加え、次のコマンド群が使用可能になる。

* `quel_xsdb_server`: hw_server と xsdb_server を立ち上げて、xsdb_server を hw_server に conn した状態にする。
* `quel_xsdb_jtaglist`:  指定の xsdb_server に接続し、その xsdb_server が握っているJTAGアダプタの一覧を表示する。
* `quel_xsdb_jtagterminal`: 指定の xsdb_server に接続し、指定のアダプタIDのターミナルを結びつけたソケットのポート番号を取得する。
* `quel_hw_server`： `xsct_server` 内部で用いられている `hw_server`の立ち上げ機能を単体で取り出したコマンド。

### シナリオ１の解決法
「環境構築」にて作成した仮想環境内で次のようにする。
```shell
source /tools/Xilinx/Vivado/2019.1/settings64.sh
quel_xsdb_server --target_type cmod --xsdb_port 36335 --hwsvr_port 6121
```

別のターミナルから、次のようにすると 接続されている Cmod A7の一覧が得られる。
```shell
quel_xsdb_jtaglist --xsdb_port 36335
```
とすると、
```text
2023-09-26 18:30:17,893 root       2  Digilent Cmod A7 - 35T 210328B991FCA
2023-09-26 18:30:17,893 root          3  xc7a35t (idcode 0362d093 irlen 6 fpga)
```
という風にJTAG IDが分かるので、これを使ってjtagterminalをソケットに繋ぎ、そのポートを得るには次のようにする。
```shell
quel_xsdb_jtagterminal --adapter_id 210328B991FCA
```
とすると、
```text
45717
```
という風に、ポート番号が得られる。

あとは、telnet でこのポートにつなぐとよい。
```shell
telnet localhost 45717
```

あるいは、エラーハンドリングが雑でよければ、
```shell
telnet localshot $(xsct_jtagterminal --adapter_id 210328B991FCA)
```
などとすることもできる。

同様に、別のターミナルにて、2020.1 の hw_server を立ち上げるには、`quel_xsdb_server`コマンドを使ってもよいし
ミニマルにやりたければ、`quel_hw_server` を使ってもよい。
```shell
quel_hw_server --hwsvr_port 4121 --target au50
```
既にデフォルトポートの3121は使われてしまっているので、別のポートを明示的に指定する必要がある。
あとは、このポート4121に対して、書き込みスクリプトを実行すればよい。

実際には、このハードウェアサーバが握っているアダプタの一覧などみたいことが多いので、xsct_server を使っておいた方
が便利である。その場合には、次のようにする。
```shell
quel_xsdb_server --hwsvr_port 4121 --target au50 --xsdb_port 34335
```
xsdb_server のデフォルトポートである 33335 も既に使用されてしまっているので、指定が必要である。

こうすれば、`xsct_jtaglist`コマンドでアダプタ一覧を取得できる。
このときに、接続する xsdb_serverのポートを指定しないと、デフォルトポートを握っている 2019.1 の xsdb_server に繋がってしまう。
```shell
quel_xsdb_jtaglist --port 34335
```
正しく Alveo-DMBv1 だけが表示されることを確かめられるはずである。
下の例は Alveo-DMBv1 を2つ繋いでいる場合である。
ひとつ(末尾　C2AA) はアダプタ単体、もうひとつのアダプタ(末尾 SBAA)には 電源の入ったAlveo U50が繋がっている。
```text
2023-09-26 18:47:29,300 root       3  Xilinx Alveo-DMBv1 FT4232H 500202A50C2AA (error DR shift through all ones)
2023-09-26 18:47:29,301 root       4  Xilinx Alveo-DMBv1 FT4232H 500202A50SBAA
2023-09-26 18:47:29,301 root          5  xcu50 (idcode 14b77093 irlen 12 fpga)
2023-09-26 18:47:29,301 root             6  bscan-switch (idcode 04900101 irlen 1 fpga)
2023-09-26 18:47:29,301 root                7  unknown (idcode 04900220 irlen 1 fpga)
```

### API
quel_pyxsdbパッケージは、`quel_xsdb_jtagterminal`コマンドに相当する機能の `get_jtagterminal_port()`関数を提供している。
これを使うと、同様のことを Pythonスクリプト内から行うこともできる。
つまり、Microblaze debug module(MDM)内のUARTを介して、FPGA上のCPUにコマンドを送ったり、テレメトリを取得したりして、様々な
実験を自動化できる。

### シナリオ２の解決法
シナリオ2では、書き込み先のアダプタIDは既に分かっているはずなので、my_hw_server を使うのがよいだろう。
ただし、これでは通常の方法と大差ない。
近い将来に、[quel_staging_tool](../quel_staging_tool) に hw_server を自動立ち上げする機能を追加する予定である。
そうなると、一台のPCで簡単にファームウェアの並列アップデートができるようになるはずだ。
