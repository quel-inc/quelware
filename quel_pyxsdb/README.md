# quel_pyxsdb ことはじめ

## なにをしてくれるパッケージなのか？
- Vivado/Vitis の xsdb の機能を使った自動化スクリプトを書くための支援ライブラリ。
    - Python API （`XsctClient` クラス） を提供。
- 複数の異なるバージョンのVivado/Vitis の xsdbサーバを共存させる仕組み提供。 
- Vivado/Vitis をリモートから使用する枠組みの提供。
    - 任意のバージョンのVivado/Vitis の xsdbサーバを systemdサービスとして起動する仕組みの提供。

この文書では、以降、パッケージのビルドやsystemd サービスのインストールについて述べ、最後にCLIコマンドの使い方を説明する。
xsdbやhw_serverの詳細については [DETAILS.md](./DETAILS.md) を参照のこと。


## パッケージのビルド
github から sugita_experimental リポジトリをクローンした後に、次の手順でパッケージをビルドする。
以降では、リポジトリをクローンしたディレクトリを　${WORK} と呼ぶことにする。

```shell
cd sugita_experimental/quel_pyxsdb
python3.9 -m build
```

dist ディレクトリの中に whl パッケージができるので、それを任意の仮想環境のインタプリタにインストールすると良い。

## Systemdサービスのインストール
次に示す 4つのsystemd サービスを作成する場合の手順を示す。

|サービス名 | ターゲットボード | Vivadoバージョン  | ポート番号(xsdb) | ポート番号 (hw_server)  | 起動  |
|----|----|----|----|----|----|
|   `xsdb-au50`    | Alveo U50     | 2020.1          | 33335           | 3121          | 自動 |
|   `xsdb-au200`   | Alveo U200    | 2020.1          | 34335           | 4121          | 自動 | 
| `xsdb-exstickge` | ExStickGE     | 2022.1          | 35335           | 5121          | 手動  |
|   `xsdb-cmod`    | CMOD A7-35T   | 2019.1          | 36335           | 6121          | 自動  |

### サービス用の仮想環境の構築
/opt/quel/xsdb_venv に仮想環境を作成し、前節でビルドしたパッケージをインストールする。(X,Y,Z は適切なバージョン番号に置き換える。)

```shell
sudo mkdir -p /opt/quel
cd /opt/quel
sudo python3.9 -m venv xsdb_venv

sudo -i
cd /opt/quel
source xsdb_venv
pip install -U pip
pip install ${WORK}/sugita_experimental/quel_pyxsdb/dist/quel_pyxsdb-X.Y.Z-py3-none-any.whl
deactivate
exit
```

### サービスの登録
/etc/systemd/system 以下に、${WORK}/sugita_experimental/pyxsct/service.examples 以下の4つのファイルをコピーし、足りない情報を追加する。

```
[Unit]
Description="Xsdb server for Alveo U50 based on Vivado 2020.1"

[Service]
User= **ここを埋める**
Group= **ここを埋める**
WorkingDirectory=/opt/quel/xsdb_venv/
ExecStart=/opt/quel/xsdb_venv/bin/quel_xsdb_server --xsdb_port 33335 --hwsvr_port 3121 --target_type au50 --vivado_topdir /tools/Xilinx/Vivado/2020.1
Restart=on-failure

[Install]
WantedBy=multi-user.target
```
User と Group の中身に、hw_server の実行に必要な権限をもったユーザのユーザ名とグループ名を書く。
ubuntuでは、hw_serverを使うためにはsudoグループに入っている必要があるようだ。
おそらく、dialoutにも所属している方がよいだろう。
ちゃんとやるのであれば、このサービス用のユーザを作るのが好ましいのは言うまでもない。

また、これらのserviceの定義では、Vivado と Vitis あるいは XSDK が標準の `/tools/Xilinx` にインストールされている
想定なので、必要に応じて調整していただきたい。

### サービスの登録の確認
次のようにして、前節で作成したサービスの状態を確認する。

```shell
sudo systemctl status xsdb-au50
```
として、
```shell
● xsdb-au50.service - "Xsdb server for Alveo U50 based on Vivado 2020.1"
     Loaded: loaded (/etc/systemd/system/xsdb-au50.service; disabled; vendor preset: enabled)
     Active: inactive (dead)
```
などと出ればOK。

### サービスの有効化
Alveo U50, Alveo U200, 及び CMOD の3つのサービスについては、次のようにしてホストの起動時に自動で xsdbが起動するようにする。
```shell
sudo systemctl enable xsdb-au50
sudo systemctl enable xsdb-au200
sudo systemctl enable xsdb-cmod
```

次にホストを再起動した際には、サービスが自動で起動されるが、今回だけはそうでないので手動で起動する。
```shell
sudo systemctl start xsdb-au50
sudo systemctl start xsdb-au200
sudo systemctl start xsdb-cmod
```

ExStickGEのxsdb は自動起動をしないことにしたので、必要になる度に都度、次のコマンドで手動で起動することにする。
```shell
sudo systemctl start xsdb-exstickge
```
使い終わったら、次のコマンドでサービスを止める。
```shell
sudo systemctl stop xsdb-exstickge
```

## CLI コマンド
### アダプタのリスト取得
Alveo U50 のJTAG アダプタがホストのUSBポートに繋がっている場合には、次のコマンドで確認できる。
```shell
quel_xsdb_jtaglist --xsdb_port 33335
```

たとえば、次のような出力が得られる。
```text
2023-11-28 18:39:25,650 root       8  Xilinx Alveo-DMBv1 FT4232H 500202a50nhAA
2023-11-28 18:39:25,650 root          9  xcu50 (idcode 14b77093 irlen 12 fpga)
2023-11-28 18:39:25,650 root            10  bscan-switch (idcode 04900101 irlen 1 fpga)
2023-11-28 18:39:25,650 root               11  unknown (idcode 04900220 irlen 1 fpga)
2023-11-28 18:39:25,650 root      12  Xilinx Alveo-DMBv1 FT4232H 500202A50JVAA
2023-11-28 18:39:25,650 root         13  xcu50 (idcode 14b77093 irlen 12 fpga)
2023-11-28 18:39:25,650 root            32  bscan-switch (idcode 04900101 irlen 1 fpga)
2023-11-28 18:39:25,650 root               33  unknown (idcode 04900220 irlen 1 fpga)
```

２つのアダプタ (500202a50nhAA と 500202A50JVAA)が繋がっていることが分かる。

`quel_xsdb_jtaglist` は、`--host` と `--xsdb_port` および　`--hwsvr_port` の3つの引数を与えると、リモートホスト上の xsdb_server を介して、リモートホストに接続している
FPGAの一覧の確認できる。
`--host` を省略すると、ローカルホストの指定のポートへアクセスすることになる。
また、systemd に登録されている4つのサービスで用いている xsdbのポート番号とhw_serverのポート番号の組み合わせについては、
`--hwsvr_port` を省略できる。
また、`--xsdb_port`を省略すると、33335番ポートの接続を試みる。

### コンソールの取得
Microblaze Debug Module (MDM)が提供するコンソールへアクセスするためのソケットを作成できる。
たとえば、ローカルホストのUSBにCMOD A7-35T のJTAGアダプタ(210328B7915DA) が接続されている状況を想定しよう。
xsdb_cmod のサービスが動作していれば、JTAGアダプタの存在を次のコマンドで確認できる。
```bash
quel_xsdb_jtaglist --xsdb_port 36335
```
コマンドの出力から、JTAGアダプタの存在が確認できる。
```text
2023-11-29 17:41:49,983 root      11  Digilent Cmod A7 - 35T 210328B7915DA
2023-11-29 17:41:49,983 root         12  xc7a35t (idcode 0362d093 irlen 6 fpga)
```

ここで、次のコマンドでこのJTAGアダプタが公開しているコンソールと接続するためのソケットを取得する。
```bash
quel_xsdb_jtagterminal --xsdb_port 36335 --adapter 210328B7915DA
```
このコマンドが返すソケットに対して、`telnet`で接続をすればよい。
たとえば、コマンドが `34143` を返した場合には、次のようにしてコンソールへ接続する。
```bash
telnet localhost 34143
```
また、`quel_xsdb_jtagterminal`コマンドに、`--host`オプションを指定した場合には、ソケットもそのホスト上
に作成されるので、`telnet`で`localhost`に接続する代わりに、そのホストへ接続する必要がある。


## 関連パッケージ
### [quel_staging_tool](../quel_staging_tool/README.md)
Alveo U50 や ExStickGE へのファームウェア書き込みと、FPGAのJTAGアダプタ経由でのリセットに対応している。
本ライブラリで立ち上げた hw_server を介して、各種動作を実行する。
特に複数の異なるバージョンのvivadoを使い分けるのに、本ライブラリが有用である。
将来的には、ファームウェアの複数のボックスへの同時書き込みの対応にも、本ライブラリを使用することになるはず。

### [quel_cmod_scripting](../uart_monitor/py/)
CMOD上で動作するソフトウェアコンソールを介して、ハードウェアを制御するスクリプトを書くためのライブラリ。
接続対象のボードのJTAGアダプタのIDから、コンソールに接続するためのソケットを取得するのに、本ライブラリを使用している。

### [quel_ic_config](../quel_ic_config/README.md)
quel_phase_log コマンドが温度状態の取得に使用している。
