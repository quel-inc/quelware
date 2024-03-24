# quel_ic_config を使ってみる
[quelwareリポジトリ](https://github.com/quel-inc/quelware)を取得するところから、CLIコマンドやハイレベルAPIの基本的な動作を確認するまでを説明する。

## 環境構築
### 仮想環境の作成
任意の作業ディレクトリで以下の手順を実行すると、新しい仮想環境が利用可能な状態になる。
```shell
python3.9 -m venv test_venv
source test_venv/bin/activate
pip install -U pip
```

### リポジトリの取得
リポジトリのクローンがまだであれば、適当な作業ディレクトリに移動の後、次のコマンドで本リポジトリをクローンし、quel_ic_config ディレクトリに移動する。
```shell
git clone git@github.com:quel-inc/quelware.git
cd quelware/quel_ic_config
```

### コンパイル済みパッケージの取得と展開
次のコマンドを実行すると、 コンパイル済みのパッケージをリポジトリの[リリースページ](https://github.com/quel-inc/quelware/releases/)から自動で取得する。
```shell
./download_prebuilt.sh
tar xfv quelware_prebuilt.tgz  
```

`wget` コマンドが無い場合には事前にインストールするか、手動でのダウンロードをすることになる。
```shell
sudo apt install wget
```

### インストール
ダウンロードしたパッケージを展開すると、その場に `requirements_*.txt` ファイルを始めとするいくつかのファイルとディレクトリができる。
制御装置のファームウェアの種類や使用目的によって、インストール時に参照する `requrements._*.txt` が異なる。
ファームウェアの種類は以下の3つである。

| ファームウェアの種類の名前　       | 概略                                                                                                     | 
|----------------------|--------------------------------------------------------------------------------------------------------|
| SIMPLEMULTI_CLASSIC  | 20240125より前のsimplemulti版ファームウェア <br> QuBE 及び QuEL-1 の出荷時ファームウェア。<br> SIMPLEMULTI_STANDARDへのアップグレードを推奨。 |
| SIMPLEMULTI_STANDARD | 20240125以降のsimplemulti版ファームウェア <br> QuEL-1 SE 及びNEC様向けモデルの標準ファームウェア                                    |
| FEEDBACK             | フィードバック研究用の実験なファームウェア（特定ユーザ様専用）                                                                        |

なお、装置にインストールされているファームウェアの種類は次のコマンドで確認できる。
```text
helpers/detect_firmware_type.sh 10.1.0.xxx
```
現状では、ファームウェアの種類の異なる装置を1つの仮想環境から使用することができない。
この制限は、近い将来に撤廃される予定である。

#### `SIMPLEMULTI_CLASSIC`の場合
この文書を執筆している時点で、ほぼ全ての QuBEおよびQuEL-1は、この状態に相当すると思う。

次のコマンドでパッケージをインストールする。
```shell
pip install -r requirements_simplemulti_classic.txt
```

ファームウェアアップデートツールなどの開発用のツール群もインストールしたい場合には、次のようにする。
```text
pip install -r requirements_simplemulti_classic.txt -r requirements_dev_addon.txt
```

#### `SIMPLEMULTI_STANDARD`の場合
次のコマンドでパッケージをインストールする。
```shell
pip install -r requirements_simplemulti_standard.txt
```

ファームウェアアップデートツールなどの開発用のツール群もインストールしたい場合には、次のようにする。
```text
pip install -r requirements_simplemulti_standard.txt -r requirements_dev_addon.txt
```

#### Feedback版のファームウェアを利用する場合（オプション）
feedback版のファームウェアはベータ版として配布しているが、時刻同期周辺が安定版のファームウェア(simplemulti版)と互換性がない。
QuEL-1 の装置詳細を理解している研究目的のユーザ以外は使用するべきでない。

次のコマンドでパッケージをインストールすればよい。
```shell
pip install -r requirements_feedback.txt
```

ファームウェアアップデートツールなどの開発用のツール群もインストールしたい場合には、次のようにする。
```text
pip install -r requirements_feedback_classic.txt -r requirements_dev_addon.txt
```

### quel_ic_config の再ビルド（オプション）
ビルド済みパッケージを使用することを推奨するが、何からの理由でquel_ic_config の再ビルドをしたい場合には、次の手順で行える。

パッケージの作成には[buildパッケージ](https://pypi.org/project/build/)を使う。
```
pip install build
python -m build
```
パッケージファイルは、`dist/quel_ic_config-X.Y.Z-cp39-cp39-linux_x86_64.whl` (X,Y,Z は実際にはバージョン番号になる) という名前で作成される。
バージョン番号を振り直した場合には、`requirements*.txt` の内容と齟齬が生じるので注意して頂きたい。

## シェルコマンドを使ってみる
quel_ic_config のパッケージにはいくつかの便利なシェルコマンドが入っており、仮想環境から使用できる。
仮に、10.1.0.xxx のIPアドレスを持つ制御装置（QuEL-1 Type-A)をターゲットとして説明するが、 IPアドレスと制御装置モデルを各自のものに合わせれば、そのまま実行可能であるはずだ。
制御装置モデルは、`--boxtype`引数の識別子として指定する。

識別子の一覧は以下のとおりである。
各モデルの詳細は[README.md](./README.md)を参照いただきたい。

| モデル名                 | 識別子              | 出荷番号（ブロック番号-個体番号)               |
|----------------------|------------------|---------------------------------|
| QuEL-1 最初期型          | `qube-riken-a`   | QuEL-1 #1-xx                    |
| QuEL-1 標準型 タイプA機     | 　`quel1-a`　      | QuEL-1 #2-xx, #3-xx, #5-xx, #6-xx |
| QuEL-1 標準型 タイプB機     | `quel1-b`        | 同上                              | 
| QuEL-1 7GHzモデル       | `quel1-a`        | QuEL-1 #4-xx                    |
| QuEL-1 NECモデル        | `quel1-nec`      | QuEL-1 #7-xx                    |
| QuEL-1 SE Riken-8モデル | `quel1se-riken8` | QuEL-1 SE #1-xx, #2-xx          |
| QuBE OU タイプA機        | `qube-ou-a`      | QuBE OU #1-xx, #2-xx, #3-xx     | 
| QuBE OU タイプB機        | `qube-ou-b`      | 同上                              | 
| QuBE Riken タイプA機     | `qube-riken-a`   | QuBE Riken #1-xx                | 
| QuBE Riken タイプB機     | `qube-riken-b`   | 同上                              | 

タイプAとタイプBの識別は、本体背面パネルに貼ってあるシールを確認していただきたい。
タイプB機はブロック番号に続く個体番号が3の倍数である場合が多いが、例外（たとえば、QuBE OU #3-02はタイプB）もあるので注意が必要だ。

### データリンク状態の確認
何か障害が発生したときに、まず装置のDAC/ADCのデータリンク状態を確認したくなるだろうと思う。
次のコマンドで状態を確認できる。

```shell
quel1_linkstatus --ipaddr_wss 10.1.0.xxx --boxtype quel1-a
```

装置が正常な運用状態にあれば、次のような出力を得る。
```text
AD9082-#0: healthy datalink  (linkstatus = 0xe0, error_flag = 0x01)
AD9082-#1: healthy datalink  (linkstatus = 0xe0, error_flag = 0x01)
```
`linkstatus` の正常値が0xe0であること、`error_flag`の正常値が0x01 であることを覚えておいて頂きたい。

装置の電源や接続状態に問題がある場合には、次のような出力を得る。
```text
2024-03-22 10:46:18,380 [ERRO] nullLibLog: timed out,  Dest ('10.1.0.xxx', 16385)
2024-03-22 10:46:18,380 [ERRO] quel1_linkstatus: timed out,  Dest ('10.1.0.xxx', 16385)
2024-03-22 10:46:18,429 [ERRO] root: cannot access to the given IP addresses 10.1.0.xxx / 10.5.0.xxx
```
ログからは通信に失敗していることが読み取れる。
可能な原因として、電源未投入、ネットワーク未接続、クロック分配器の電源未投入、あるいは、クロックケーブル未接続などがある。
機器状態の再確認のうえ、リトライしていただきたい。

初期化（いわゆるリンクアップ）が済んでいない場合には、次のような出力を得る。
```text
2023-11-07 17:54:29,111 [ERRO] quel_ic_config.ad9082_v106: Boot did not reach spot where it waits for application code
2023-11-07 17:54:29,111 [ERRO] quel_ic_config_utils.basic_scan_common: failed to establish a configuration link with AD9082-#0
2023-11-07 17:54:29,111 [ERRO] quel_ic_config_utils.basic_scan_common: AD9082-#0 is not working. it must be linked up in advance
2023-11-07 17:54:29,330 [ERRO] quel_ic_config.ad9082_v106: Boot did not reach spot where it waits for application code
2023-11-07 17:54:29,330 [ERRO] quel_ic_config_utils.basic_scan_common: failed to establish a configuration link with AD9082-#1
2023-11-07 17:54:29,330 [ERRO] quel_ic_config_utils.basic_scan_common: AD9082-#1 is not working. it must be linked up in advance
AD9082-#0: no datalink available  (linkstatus = 0x00, error_flag = 0x00)
AD9082-#1: no datalink available  (linkstatus = 0x00, error_flag = 0x00)
```
総合的な判定は、最後の2行にある。
正常な場合では0xe0だった`linkstatus` が、0x00 になっている。
ログ表示が少々鬱陶しいが、ログ情報はトラブルシューティングに有用なので敢えて表示している。

初期化に失敗した状態で放置するなどして、異常が発生している場合には、`link_status`が 0x90 や 0xa0 など、0xe0以外の値となることで分かる。
この場合にも、再度リンクアップする必要がある。

`link_status`が正常値 (0xe0) だが、内部のデータリンクにビットフリップが発生が過去にあった場合には、`error_flag` が 0x11 になる。
次の出力例は、片側のAD9082でCRCエラーが検出されたことを示す。
```text
2024-03-22 11:20:52,333 [ERRO] quel_ic_config.quel1_box_intrinsic: AD9082-#1 is not working. it must be linked up in advance
AD9082-#0: healthy datalink  (linkstatus = 0xe0, error_flag = 0x01)
AD9082-#1: unhealthy datalink (CRC errors are detected)  (linkstatus = 0xe0, error_flag = 0x11)
```
この異常は無視しても問題ないケースがほとんどである。
納品前の検査で、 量子実験の観点で測定データに実験に影響があるようなデータ破損が無いことを確認済みである。
フラグが立っていることが気になるようであれば、折を見て再リンクアップすることをお勧めする。
とはいえ、再リンクアップから数分以内にCRCエラーのフラグが立つ事象が頻繁に発生する場合には、ご一報いただきたい。

CRCエラーを無視したい場合には、`--ignore_crc_error_of_mxfe 0,1` オプションを適用すればよい。
このコマンドをシェルスクリプト中で用いる際には、　無視する挙動が好ましいケースが多いと考える。
というのは、デフォルト動作は、CRCエラーも他の重篤な問題と同様に扱うので返り値が非ゼロとなり、bashなどで`set -e`をしている場合にはスクリプトがエラーで停止してしまう。
このオプションで、この問題を避けることができる。
目視での確認でCRCエラーが煩いと感じる場合も同様である。

### 装置の再初期化（リンクアップ）
#### 基本的な使い方
次のコマンドで、指定の制御装置の初期化ができる。
```shell
quel1_linkup --ipaddr_wss 10.1.0.xxx --boxtype quel1-a
```
初期化に成功した場合には、次のようなメッセージが表示される。
```text
ad9082-#0 linked up successfully
ad9082-#1 linked up successfully
```
この状態で、さきほどの `quel1_linkstatus` コマンドを使用すると正常状態を示す出力が得られるはずだ。

##### SIMPLEMULTI_CLASSICの場合
このコマンドは、デフォルトではType-AとType-Bの両方の機体で差し障りのないように、モニタ系が使用可能な状態に初期化を行う。
Type-Aの機体でリード系を使うように初期化をする場合には、次のようにする。
```shell
quel1_linkup  --ipaddr_wss 10.1.0.xxx --boxtype quel1-a --config_options use_read_in_mxfe0,use_read_in_mxfe1
```
Type-Aのときにはリード系をデフォルトにすることも考えたのだが、敢えて愚直な作りにしている。

##### SIMPLEMULTI_STANDARDの場合
ファームウェアに SIMPLEMULTI_STANDARD版のものを使っている場合には、リードとモニタの両系を同時に使えるので、--config_options以下は不要になる。
```shell
quel1_linkup  --ipaddr_wss 10.1.0.xxx --boxtype quel1-a
```
とすればよい。

#### リンクアップ結果の確認
リンクアップの最中に次のような警告が何行か表示されることがあるが、最後に `linked up successfully` が出ていれば問題はない。
```text
2023-11-07 19:13:13,821 [WARN] quel_ic_config_utils.quel1_wave_subsystem: timeout happens at capture units CaptureUnit.U4, capture aborted
2023-11-07 19:13:22,718 [WARN] quel_ic_config.quel1_config_subsystem: failed to establish datalink between 10.5.0.xxx:AD9082-#1 and FPGA
```

リンクアップに失敗した場合には、警告が少なくとも10行程度表示された後に、次のようなエラーが出る。
```text
2023-11-07 19:54:28,830 [ERRO] root: ad9082-#0 failed to link up
2023-11-07 19:54:28,830 [ERRO] root: ad9082-#1 failed to link up
```
リンクアップが失敗を繰り返す場合には、警告ログの内容と共に連絡を頂きたい。
コマンドを`--verbose`オプション付きで実行すると、さらに詳細な情報が得られる。
このログを頂ければ、対応検討の返答までの時間の短縮が期待できる。
なお、`--use_204c` オプションを付けると症状の改善を期待できるので、是非、試していただきたい。
このオプションは2023年度納品機（QuEL-1 SE）ではデフォルトでの適用になることが決まっている。

#### ハードウェアトラブルの一時的回避
##### CRCエラー
`quel1_linkup`コマンドは、リンクアップ中に発生し得るハードウェア異常を厳格にチェックすることで、その後の動作の安定を担保するが、
時に部分的な異常を無視して、装置を応急的に使用可能状態に持ち込みたい場合には邪魔になる。
このような状況に対応するために、いくつかの異常を無視して、動作を続行するためのオプションを用意した。
あくまで応急的な問題の無視が目的なので、スクリプト内にハードコードして使うようなことは避けるべきである。

- CRCエラー
  - FPGAから送られてきた波形データの破損をAD9082が検出したことを示す。`(linkstatus = 0xe0, error_flag = 0x11)` でリンクアップが失敗することでCRCエラーの発生が分かる。
  - `--ignore_crc_error_of_mxfe` にCRCエラーを無視したい MxFE (AD9082) のIDを与える。カンマで区切って複数のAD9082のIDを与えることもできる。
  - 上述のとおり、日に1回程度の低い頻度で発生している分には実験結果に分かるような影響はない。
  　- QuEL-1 以降では、linkup中にCRCエラーが発生する機体を出荷停止にしているので、必要ないはずである。 
  - 頻繁に発生して気になる場合には、quel1_linkup コマンドに `--use_204c`オプションを付けると発生を低減できる可能性が高い。
    - QuEL-1 SE以降ではデフォルトで `--use_204c` が適用となる。QuEL-1以前の機体にはリンクアップの成功率が下がる個体があるかもしれないので、デフォルトにしていない。 
- ミキサの起動不良
  - QuBEの一部の機体において、電源投入時にミキサ(ADRF6780)の起動がうまくいかない事象が知られている。`unexpected chip revision of ADRF6780[n]`(n は0~7) で通信がうまくいっていないミキサのIDが分かる。   
  - `--ignore_access_failure_of_adrf6780` にエラーを無視したいミキサのID (上述のn)を与える。

これらの一部は他のコマンドと共通である。他のコマンドへの適用可否については、各コマンドの`--help`を参照されたい。
これら以外にも、`ignore`系の引数がいくつかあるが、通常運用で使用するべきでない。

##### 標準より大きいADCの背景ノイズ
リンクアップ手順の最後に、各ADCについて、無信号状態での読み値に大きなノイズが乗っていないことの確認をしている。
QuEL-1の各個体については、キュエル株式会社がノイズが既定値(=256)よりも十分に小さいことを確認後に出荷しているが、QuBEについてはノイズが大きめの機体が存在する。

```text
max amplitude of capture data is XXXXX (>= 256), failed to linkup
```

のようなメッセージが5回以上繰り返し出て、リンクアップに失敗する場合には、`--background_noise_threshold` 引数で上限値を引き上げられる。
目安としては、400くらいまでが個体差の範疇であると考えてよい。
それよりも大きい値で失敗を繰り返す場合には、なんらかのハードウェア的な問題を示唆するので、装置のパワーサイクルを行い、それでも回復しない場合にはサポート窓口へ連絡して頂きたい。

なお、リンクアップ中に30000以上の値が数回出た後に、リンクアップに成功するのは既知の事象であり、装置の使用上問題はない。
初期化時のタイミングに依存した異常で、確率的に発生するが、一旦、異常なしの状態に持ち込めば、その後は安定動作する。
なお、`quel1_linkup`コマンドは異常が発生しなくなるまでリンクアップを自動で繰り返す。

#### 分かりにくい警告メッセージについて 
QuBE-RIKENの制御装置を使用する際に `--boxtype` を `quel1-a` や `quel1-b` と誤って指定した後に、正しく `qube-riken-a` や `qube-riken-b` に
指定し直した場合に、次のような警告メッセージが出ることがある。
```
invalid state of RF switch for loopback, considered as inside
```

`quel1_linkup` でこのメッセージが出る場合、つまり、当該装置を間違った boxtype で使用した後で、正しい boxtype で再リンクアップを試みた場合には、無害なので無視してよい。
なぜならば、`quel1_linkup`コマンドがスイッチの状態を初期化するからである。
他の状況でこのメッセージが出る場合には、すべてのRFスイッチの状態を再設定しておいた方がよいだろう。


### 装置の設定状態の確認
各ポートの設定パラメタの一覧を次のコマンドで確認できる。
```shell
quel1_dump_port_config --ipaddr_wss 10.1.0.xxx --boxtype quel1-a
```

全てのポートのについて、次のような情報が表示される。
```text
{'mxfes': {0: {'channel_interporation_rate': 4, 'main_interporation_rate': 6},
           1: {'channel_interporation_rate': 4, 'main_interporation_rate': 6}},
 'ports': {0: {'direction': 'in',
               'lo_freq': 8500000000,
               'cnco_freq': 1999999999.9999716,
               'rfswitch': 'open',
               'runits': {0: {'fnco_freq': 0.0},
                          1: {'fnco_freq': 0.0},
                          2: {'fnco_freq': 0.0},
                          3: {'fnco_freq': 0.0}}},
           1: {'direction': 'out',
               'channels': {0: {'fnco_freq': 0.0}},
               'cnco_freq': 1500000000.0,
               'fullscale_current': 40527,
               'lo_freq': 8500000000,
               'sideband': 'U',
               'rfswitch': 'pass'},
           2: {'direction': 'out',
               'channels': {0: {'fnco_freq': 0.0},
                            1: {'fnco_freq': 733333333.3333272},
                            2: {'fnco_freq': 766666666.6666657}},
               'cnco_freq': 1500000000.0,
               'fullscale_current': 40527,
               'lo_freq': 11500000000,
               'sideband': 'L',
               'rfswitch': 'block'},
...,
}
```

ただし、取得する方法がない出力ポートのVATTの値だけは表示されないので注意が必要である（このコマンドの実装に用いているAPIは、おなじ実行コンテキストで設定されたVATT値をキャッシュして返すが、コマンド自身がVATTの値を設定することはないので、何も値が得られない）。

## APIを使ってみる
次のコマンドで、装置の簡単な動作確認をインタラクティブに実施できる。
あくまでお試し用なので複雑なことをするのには向いていない。
リポジトリの`quel_ic_config`ディレクトリに移動後、次のコマンドをすることで制御装置を抽象化したオブジェクト`box`が利用可能な状態になった pythonのインタラクティブシェルが得られる。

```shell
python -i scripts/getting_started_example.py --ipaddr_wss 10.1.0.xxx --boxtype quel1-a 
```

たとえば、`box.dump_box()` とすることで、上述の `quel1_dump_port_config` コマンドの出力同様の結果を含んだデータ構造を得られる。
以下に APIの一覧を項目ごとに列挙する。
詳しい使い方は、`help(box.dump_box)` のように `help`関数で見られたい。

- 全体設定
  - **dump_box**: 制御装置全体の設定状態をデータ構造として得る。
  - **config_box**: 制御装置全体の設定を一括して行う。dump_boxで取得したデータ構造の "ports" の部分と同じデータ構造の全部か一部かを与える。

- ポート設定
  - **dump_port**: 各ポートの設定状態をデータ構造として得る。
  - **config_port**: 指定のポートのパラメタを設定する。

- RFスイッチ
  - **dump_rfswitches**: 全てのポートのRFスイッチの状態を取得する。
  - **config_rfswitches**: 全てのポートのRFスイッチの状態を設定する。
  - **block_all_output_ports**: 全ての出力ポートのRFスイッチをblock状態にする。ただし、モニタアウトは含まない。なお、以下の4つのAPIは頻繁なユースケース向けに config_rfswitches を特化したものである。 
  - **pass_all_output_ports**: 全ての出力ポートのRFスイッチをpass状態にする。ただし、モニタアウトは含まない。
  - **activate_monitor_loop**: モニタ系のRFスイッチをループバック状態にする。
  - **deactivate_monitor_loop**: モニタ系のRFスイッチのループバックを解除する。

- 信号入出力簡易API
  - **easy_start_cw**: 指定の出力ポートから信号を出力する最も簡単な方法。お試し用のAPIであり、実用には向かない。
  - **easy_stop**: 指定の出力ポートの全ての信号発生を停止する。お試し用APIであるが、実用にもなる。
  - **easy_stop_all**: 全ての出力ポートの全ての信号発生を停止する。お試し用APIであるが、実用にもなる。
  - **easy_capture**: 指定の入力ポートの信号をキャプチャする最も簡単な方法。お試し用のAPIであり、実用には向かない。

- 信号出力API（本文書では説明しない）
  - **initialize_awgs**: 全てのAWGを初期化する。
  - **load_iq_into_channel**: AWGに任意の波形データを設定する。
  - **load_cw_into_channel**: AWGに連続派の波形データを設定する。load_iq_into_channel を連続波だけに特化したAPIである。
  - **start_emission**: ひとつ、あるいは、複数のAWGを指定して、波形発生を開始する。
  - **stop_emission**: ひとつ、あるいは、複数のAWGを指定して、波形発生を停止する。
  - **simple_capture_start**: ひとつのポートのひとつ、あるいは、複数のrunitを指定して、一回の波形キャプチャの開始トリガの設定、あるいは、即時の開始をする。キャプチャ波形の Futureオブジェクトを返す。
  - **capture_start**: Boxでは未実装。要望があれば実装優先度を高める。ひとつのポートのひとつ、あるいは、複数のrunitを指定して、複数回の波形キャプチャの開始トリガの設定をする。取得したキャプチャ波形を順次取り出せる Futureオブジェクトをラップした Iteratorオブジェクトを返す。

まずは、 Read-outポートから信号発生し、RFスイッチを閉じた状態にして内部ループバックの経路を有効化して、Read-inから信号を読んでみる。

FY2022納品のType-A機(`quel1-a`)では次の手順で確認できる。
このモデルでは、0番ポートがRead-in、1番ポートがRead-out である。
```python
box.config_port(port = 1, lo_freq = 11.5e9, cnco_freq = 1.5e9, sideband = "L", vatt = 0xa00)
box.config_port(port = 0, cnco_freq = 1.5e9, rfswitch="loop")
box.easy_start_cw(port = 1, channel = 0, control_port_rfswitch=False)
iq = box.easy_capture(port = 0, num_samples=256)
box.easy_stop_all()
```
とすると、iq にループバック受信した直流のベースバンド信号を取得できるはずだ。
`quel1-a`以外のモデルでは、ポート番号を次のように読み替えていただきたい。

| モデル識別子           | Read-inポート | Read-outポート | 備考                  |
|------------------|------------|-------------|---------------------|
| `quel1-a`        | 0          | 1           |                     |
| `quel1-b`        | N.A.       | N.A.        | 実施不能                |
| `quel1-nec`      | 2          | 0           | RFスイッチがないので一部の設定が不要 |
| `quel1se-riken8` | 0          | 1           |                     |
| `qube-ou-a`      | 1          | 0           | RFスイッチがないので一部の設定が不要 |
| `qube-ou-b`      | N.A.       | N.A.        | 実施不能                |
| `qube-riken-a`   | 1          | 0           |                     |
| `qube-riken-b`   | N.A.       | N.A.        | 実施不能                |

結果を確認する前に少しだけコードを解説しておく。
まず、注意が必要なのは、1番ポートと0番ポートのRFスイッチは連動していることだ。
上記の例では、2行目で 0番ポートのスイッチを操作している。
その代わりに、1行目で次のようにしても同じ結果となる。
```python
box.config_port(port = 1, lo_freq = 11.5e9, cnco_freq = 1.5e9, sideband = "L", vatt = 0xa00, rfswitch="block")
```
なお、`rfswitch=`の値が `loop`でなく`block`なのは、入力ポートと出力ポートの違いを反映している。

もうひとつ注意が必要なのは、3行目の `control_port_rfswitch=False` で、これは easy_start_cw がRFスイッチを自動で開けてしまうのを止めるための処置である。
これを忘れると、せっかく2行目で閉じたループが開いてしまう。
easy_start_cw() のデフォルト動作としてRFスイッチを開けるようになっているのは、もともとスペアナで信号を測定するケースでの使用を想定しているのが理由である。

```python
abs(iq)
```
とすると、数千程度のほぼ一定の値が256個得られると思う。
もし、100以下程度の小さい値しか得られてない場合には、無信号状態を受信しているのだと思う。
手順を間違っていると思うので、確認してみて欲しい。
たとえば、`box.dump_rfswitches()` でスイッチの状態を確認できる。
```python
{0: 'open', 2: 'pass', 3: 'pass', 4: 'pass', 5: 'loop', 7: 'open', 9: 'pass', 10: 'pass', 11: 'pass', 12: 'loop'}
```
この出力例では、ポート0が "open" であることから、内部ループバック経路の形成に失敗していることが読み取れる。
なお、ポート1が返り値に含まれていないのは、ポート0とポート1のRFスイッチが連動しているので、重複した設定値を省いているからである。
同様に、ポート7とポート8もRFスイッチが連動しているので、入力側のポート7だけを返している。
なお、信号発生・取得について想定外の事象が発生したら、RFスイッチが想定どおりの状態になっていることを最初に確認するべきである。
複数のスイッチが連動していたり、また、ユーザコードにおいても、どのタイミングで誰がRFスイッチを開け締めするか、設計上の迷いどころになりがちなので。

最後に、QuEL社の開発機で実行した場合の具体的な測定値を示す。
```python
>> import numpy as np
>> np.mean(abs(iq))
2012.3402
>> np.sqrt(np.var(abs(iq)))
29.554134
```
同じ機体で、2行目の`rfswtich="loop"`を `rfswitth="open"`などとして、ループバック経路の準備に失敗していると、平均166、標準偏差44程度になった。

## 次のステップ
波形発生の実用的APIについては、サンプルコードを読んでいただくのが理解の早道だと考える。
使い勝手にやや難はあるが、参考になると思う。
- [`general_loopback_test_update.py`](./testlibs/general_looptest_common_updated.py):  boxオブジェクトを使った信号発生と取得の例として分かりやすい。
- [`simple_scheduled_loopback.py`](./scripts/simple_scheduled_loopback.py): boxオブジェクトを使って書き直すのが間に合っていないが、タイムトリガ実行のより実践的な例として有用。内部のカウンタを使って、一定間隔で5回、キャプチャを繰り返す。対象機体（10.1.0.74がハードコードされている）の2つのモニタアウトをコンバイナを介して、グループ0のRead-inに繋いだ状態で使う。 
- [`twobox_scheduled_loopback.py`](./scripts/twobox_scheduled_loopback.py): `simple_scheduled_loopback.py`を複数の制御装置に拡張したもの。内部のカウンタを使って、2台の制御装置（10.1.0.74 と 10.1.0.58) を同期してキャプチャする。2台の制御装置の合計4つのモニタアウトをコンバイナを介して、1台目のグループ0のRead-inに繋いだ状態で使う。
- [`skew_scan.py`](./scripts/skew_scan.py): `twobox_schedued_loopback.py`の内容を、開始タイミングを1クロックずつずらしながら17回行い、波形生成のタイミング関係に与える影響を確認するサンプル。
- [`quel_measurement.py`](./quel_ic_config_cli/phase_measurement.py): モニタループバックを介した出力信号の長期間位相計測を行うため非公式コマンドである quel1_phase_log のソースコード。

なお、boxオブジェクトのAPIのより詳しい情報は、[移行ガイド](./MIGRATION_TO_0_8_X.md) に記載がある。
また、[ソースコード](./quel_ic_config/quel1_box.py) の pydoc にも説明がある。
