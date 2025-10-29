# quel_ic_config の概要

quel_ic_configは、既存のキュエル社製制御装置モデルにおけるadi_api_mod および qubelsi 相当の機能の高信頼化と、
新制御装置モデルへの容易な適合を意図した、新しいホスト側ソフトウェアである。
複数の制御装置を統合するためのシステムソフトウェア開発の障害を解消し、また、将来の拡張性を確保することも開発目的である。
とりあえず使ってみる場合には、本ドキュメントを参照いただきたい。
また、[scriptsディレクトリ内](../scripts)のサンプルスクリプトの使用法については、[こちらのドキュメント](./HOW_TO_USE_EXAMPLES.md)を参照いただきたい。

- QuEL-1のリンク確立、設定状態の取得、簡単な動作試験をするためのコマンド群を提供。
    - `quel1_linkup`:  リンク確立手順を実施する。
    - `quel1_linkstatus`:  現在のリンク状態を表示する。
    - `quel1_parallel_linkup`: 設定ファイルに記述した全ての装置に対し、同時にリンク確立手順を実施する。
    - `quel1_sync`: 設定ファイルに記述した全ての装置を、指定のクロックマスタを用いて時計合わせをする。
    - `quel1_syncstatus`: 設定ファイルに記述した全ての装置の時計のずれを表示する。
    - `quel1_dump_port_config`:  各ポートごとの設定状況を表示する。
    - `quel_config_ports`:  各ポートごとの設定の表示や変更を行う（新しい）ツール。
    - `quel1_firmware_version`: ファームウェアバージョンを表示する。
    - `quel1_check_all_loopbacks`: 制御装置の全出力ポートの健全性を内部ループバックで確認するツール。QuEL-1 SE RIKEN8モデル以外で使用する。
    - `quel1se_riken8_check_all_loopbacks`: 制御装置の全出力ポートの健全性を内部ループバックで確認するツール。QuEL-1 SE RIKEN8モデル限定。
    - `quel1se_tempctrl_state`: QuEL-1 SE の恒温制御の稼働状態を表示する。
    - `quel1se_tempctrl_reset`: QuEL-1 SE の恒温制御を再始動する。環境温度が大きく変動した場合に温度目標値をリセットするのに使う。

なお、いくつかのコマンドではsystem configurationファイルの用意が必要となる。
詳しくは、[こちらのドキュメント](./SYSTEM_CONFIGURATION.md)を参照いただきたい。

- QuEL-1の内部リソースを概念として組織化し、各コンポーネントの設定を容易かつ意図通りに行えるAPIを提供。
    - 次に示す階層構造にしたがって整理されており、全てのキュエル社製の制御装置を同じように扱うことができる。
      - 各コンポーネントのハードウェアを抽象化する層 (HAL)
      - 各コンポーネントをQuEL-1の機能の文脈で抽象化し直した層 (CSS)
      - 波形発生及び取得の機能をQuEL-1の機能の文脈で抽象化した層 (WSS)
      - 全てのキュエル社製制御装置モデルで共通化したAPIを提供する層 (Box) 
      - (複数の制御装置をひとつの量子計算機制御システムとして見せる層 (開発中)）
    - Boxオブジェクトを quel_ic_config.Quel1Box.create() を介して作成することで、ハードウェアの詳細に関する面倒なポイントを分かりやすく安全に使えるAPIを提供。
      - 構成の異なるポートを統一的に設定できる柔軟なAPIで、厳格な静的・動的チェックをする内部メカニズムをラッピングして提供。
      - 複数の用途で共有されているリソース設定で生じがちな矛盾を検出して、意図しない動作を未然に防ぐAPI設計。

## 対応制御装置モデル
本ライブラリがサポートしている制御装置モデルの一覧を示す。
ライブラリの初期化時に対象制御装置のモデル名を指定する必要があるので、その識別子も合わせて説明する。

### QuEL-1 (Type-A 及び Type-B)
キュエル株式会社が製造した2022年度から2023年度にかけて製造・販売した制御装置で、 以下の示すとおり、3種類のバリエーションがある。
- 最初期型: 2022年の10月以前に納品した機体。ポート配置がQuBE-RIKENと同じなので、本ライブラリでは　QuBE-RIKEN Type-A と同一視する。 
- 標準型: 2023年2月以降に納品した機体のほとんどすべて。２つの構成 Type-A と Type-B が存在。
- 7GHzモデル： 制御信号の出力帯域が標準型と異なるカスタムタイプ。ソフトウェア的には標準型と等価なので、殆どの場合、QuEL-1 Type-A と同一視できる。

### QuEL-1 NEC
4つの独立した量子ビットを制御するためのQuEL-1のカスタムモデル。

### QuEL-1 SE
QuEL-1 のハードウェアを大幅にリファインした2023年度末モデル。
恒温系ハードウェアの全面的見直し、周波数コンバータのモジュール化、設定系ファームウェアの全面改修と機能向上など、性能と構成の柔軟性の両方を大幅に向上させている。

- RIKEN 8GHzモデル: QuEL-1 SEのローンチモデルで、理研の5GHz帯量子ビットを制御するために、2-8GHzのRF入出力を持つ。
- FUJITSU 11GHzモデル: QuEL-1 Type-A/-B 標準型と同等構成の装置。

### QuBE (Type-A 及び Type-B)
QuEL-1 の元になった制御装置で、最初期型の QuBE-OU と、その改良型の QuBE-RIKEN とがある。
前者にはモニタ系が実装されていない。後者は　QuEL-1 と同様にモニタ系を内蔵している。
これらのモデルは、キュエル株式会社の納品物ではないが、本ライブラリの動作をサポートする。
後述するように、QuEL-1 とは大幅にポートの並び異なるので注意が必要である。

# 使ってみる

[quelwareリポジトリ](https://github.com/quel-inc/quelware)を取得するところから、CLIコマンドやハイレベルAPIの基本的な動作を確認するまでを説明する。
以下の手順の動作確認は、Ubuntu 20.04.6 LTS で行っているが、他のLinuxディストリビューションでも同様の手順で環境構築ができるはずだ。

## 0.10.x 系の変更点
- 今後のファームウェアの抜本的なアップデートへの対応準備と安定性向上の目的で、e7awgsw を e7awghal で置き換えた。これに伴い、いくつかの制限事項が発生する。
    - READ-IN と MONITOR-IN が同時に使えない古いファームウェアのサポートを停止。
    - Feedback実験版のファームウェアのサポートを一時的に停止させて頂いた。要望があればサポートを再開するが、キュエル社としてはフィードバック機能を再設計したファームウェア開発後に対応再開としたいと考えている。
        - それまでは、0.8.x系の quelware をご使用頂きたい。 
- WaveSubsystem を再設計したことに伴い、波形生成及び波形取得に関するAPIを大幅に変更した。詳細は[こちら](./MIGRATION_TO_0_10_X.md)。
    - WaveSubsystemのAPIはこれで固定。AD9082の500MHz情報帯域の出力チャネル（channel) をそのままユーザに見せるAPI、という位置づけ。
    - 一方で、measurement tools のバーチャルポート相当の、量子ビットや読み出し共振器の共振周波数を中心とする比較的狭帯域の位相管理された出力ユニット（tunit) に基づくインターフェースを別途開発する予定。
- 各装置の排他ロック機能が更新されている。従来は、e7awgswがファームウェアの一部について排他制御をしていたが、これを次のように改めた。
    - 1台の制御装置全体に排他制御が掛かるようにした。ある装置に対応するboxオブジェクトが存在している間は、他ユーザがその装置のboxオブジェクトを作成できなくなる。
    - QuEL-1以前の装置では、これまで通りロックファイルを用いた排他制御を行う。
        - ロックファイルの置き場を、`/run/quelware` ディレクトリに変更した。後で説明をするが、ディレクトリを作成しておく必要がある。
    - QuEL-1 SE以降については、ロックファイルを用いる代わりにデバイスそのものにロックを掛けることもできるが、ファームウェアのアップデートが必要である。
        - 各ICの設定は、他のユーザの任意のホストからの書き換えアクセスを遮断できる。
        - AU50の設定は、quelware-0.10.x 以降を使用している場合に限り、他のユーザの任意のホストからの書き換えアクセスを遮断できる。
        - デバイスロックは、ExStickGEのファームウェアをv1.3.0 以降での対応となる。
            - ExStickGEのファームウエアのOTA アップデートするツールで、簡単にアップデートを実施できる。
            - 0.10.1b3 リリースのタイミングでは、v1.3.0ファームウェアは公開していない。

## 環境構築
### グラフ描画用ライブラリのバックエンド設定
サンプルスクリプトやテストコードにて、グラフの表示に matplotlib を用いている。
場合によっては描画用のバックエンドを指定する必要がある。
詳しくは [matplotlib の公式ドキュメント](https://matplotlib.org/stable/users/explain/figure/backends.html)を参照のこと。

#### バックエンドの設定例
ユーザ設定ファイル `matplotlibrc` 上でバックエンドを指定する場合を紹介する。
設定ファイルを置くべきパスを取得するには、matplotlib がインストールされた状態で、次のコマンドを実行する。
```shell
python -c "import matplotlib as mpl; print(mpl.get_configdir())"
```
ユーザ設定としてバックエンドを指定するには、このディレクトリ内にファイル `matplotlibrc` を次の内容で作成する。
```text
backend: GTK3Agg
```
この例ではバックエンドとして `GTK3Agg` を指定している。
各種バックエンドの解説は [matplotlib の公式ドキュメント](https://matplotlib.org/stable/users/explain/figure/backends.html)を参照のこと。
それぞれのバックエンドが利用可能かどうかはシステム環境に依存しており、追加のインストールが必要となる場合がある。
例えば `GTK3Agg` をUbuntu 20.04環境で使う場合には、以下の手順でシステムにパッケージをインストールしておく必要がある。
```shell
sudo apt install libgirepository1.0-dev gcc libcairo2-dev pkg-config python3-dev gir1.2-gtk-3.0
```

### ロックファイル用ディレクトリの作成
Ubuntu-20.04 では、ブート時の自動作成の設定を `/etc/tmpfiles.d/` 以下のファイルに記述する。
まず、`/etc/tmpfiles.d/quelware-lock.conf` を次の内容で作成する。
```text
d /run/quelware 777 root root -
```
パーミッションの設定は運用ポリシーと相談であるが、全ユーザにサーバを介して制御装置にアクセスすることを強制するのでなければ、 777 や 775 に
設定するのが一般的だろう。
この設定を有効化するには、PCを再起動するか、あるいは、次のコマンドを実行する。
```shell
sudo systemd-tmpfiles --create
```

ロックの詳細やロックファイル用ディレクトリの複数ホスト間での共有については、[こちらの文章](./BOX_EXCLUSIVE_ACCESS_CONTROL.md)を参照のこと。

### Python仮想環境の作成
任意の作業ディレクトリで以下の手順を実行すると、新しい仮想環境が利用可能な状態になる。
```shell
python3.9 -m venv test_venv
source test_venv/bin/activate
pip install -U pip
```

### インストール
次のコマンドによってインストールできる。
```
pip install quel_ic_config
```

### 開発者向け依存パッケージを含めたインストール
リポジトリのクローンがまだであれば、適当な作業ディレクトリに移動の後、次のコマンドで本リポジトリをクローンし、quel_ic_config ディレクトリに移動する。
```shell
git clone git@github.com:quel-inc/quelware.git
cd quelware/quel_ic_config
```
もし古いバージョンが必要な場合は、対応するブランチにswitchすること。

そして次のコマンドを実行することで、開発者向けの依存パッケージをインストールできる。
```
pip install ../quel_pyxsdb ../quel_staging_tool ../quel_cmod_scripting quel_ic_config[dev]
```

#### ファームウェアによる差異について
v0.10以降では、全ての制御装置ファームウェアに対応した単一のパッケージを配布するので、これまでのような装置のファームウェアの種類によって、パッケージ
を選択する必要はない。
とはいうものの、現状、quelware-0.10.x が対応しているファームウェアはSIMPLEMULTI_STANDARDの1種類だけである。
古いSIMPLEMULTI_CLASSICのファームウェアはサポート中止であり、また、ベータ版配布のFEEDBACK_EARLYファームウェアは、一時的にサポートを停止している。

| ファームウェアの種類の名前　       | 概略                                                                     | 
|----------------------|------------------------------------------------------------------------|
| SIMPLEMULTI_CLASSIC  | 20240125より前のsimplemulti版ファームウェア <br> 本リリースで対応廃止となったのでファームウェアのアップデートが必要 |
| SIMPLEMULTI_STANDARD | 20240125以降のsimplemulti版ファームウェア <br> QuEL-1 SE 及びNEC様向けモデルの標準ファームウェア    |
| FEEDBACK_EARLY       | フィードバック研究用の実験なファームウェア（特定ユーザ様専用）<br> サポートを一時停止中。                       |

実験室で使用中の制御装置のほとんど全てに、SIMPLEMULTI_STARNDARD版のファームウェアがインストールされているという認識だが、
心配であれば、以下のコマンドで各装置のファームウェア情報を確認できる。

```text
quel1_firmware_version --boxtype xxxxx --ipaddr_wss 10.1.0.yyy
```

#### `SIMPLEMULTI_CLASSIC`の場合
ほとんど全ての制御装置のファームウェアがSIMPLEMULTI_STANDARDに更新された状況を鑑みて、本リリースで対応中止とした。
もし、SIMPLEMULTI_CLASSICのファームウェアをインストールしている機体は、従来の0.8.x系のquelwareで使用するか、あるいは、ファームウェアのアップデートが必要である。

#### `SIMPLEMULTI_STANDARD`の場合
上記のインストール手順に沿ってインストールしていただければよい。

#### `FEEDBACK_EARLY`の場合
FEEDBACK_EARLY版のファームウェアはベータ版として配布しているが、コマンドシーケンサ周辺がSIMPLEMULTI_STANDARDファームウェアと互換性がない。
本バージョンではサポートを一時停止しており、次期ファームウェアリリース時に対応再開を予定している。
現状のFEEDBACK_EARLYのファームウェアについては、当面の間、0.8.x を使用して頂きたい。

### quel_ic_config の再ビルド（オプション）
ビルド済みパッケージを使用することを推奨するが、何からの理由でquel_ic_config の再ビルドをしたい場合には、次の手順で行える。

パッケージの作成には[buildパッケージ](https://pypi.org/project/build/)を使う。
```
pip install build
python -m build
```
パッケージファイルは、`dist/quel_ic_config-X.Y.Z-cp39-cp39-linux_x86_64.whl` (X,Y,Z は実際にはバージョン番号になる) という名前で作成される。

## シェルコマンドを使ってみる
quel_ic_config のパッケージにはいくつかの便利なシェルコマンドが入っており、仮想環境から使用できる。
仮に、10.1.0.xxx のIPアドレスを持つ制御装置（QuEL-1 Type-A)をターゲットとして説明するが、 IPアドレスと制御装置モデルを各自のものに合わせれば、そのまま実行可能であるはずだ。
制御装置モデルは、`--boxtype`引数の識別子として指定する。

識別子の一覧は以下のとおりである。
各モデルの詳細は[README.md](../README.md)を参照いただきたい。

| モデル名                          | 識別子                   | 出荷番号（ブロック番号-個体番号)                           |
|-------------------------------|-----------------------|---------------------------------------------|
| QuEL-1 最初期型                   | `qube-riken-a`        | QuEL-1 #1-xx                                |
| QuEL-1 標準型 タイプA機              | `quel1-a`　           | QuEL-1 #2-xx, #3-xx, #5-xx, #6-xx           |
| QuEL-1 標準型 タイプB機              | `quel1-b`             | 同上                                          | 
| QuEL-1 7GHzモデル                | `quel1-a`             | QuEL-1 #4-xx                                |
| QuEL-1 NECモデル                 | `quel1-nec`           | QuEL-1 #7-xx                                |
| QuEL-1 SE Riken-8モデル          | `quel1se-riken8`      | QuEL-1 SE #1-xx, #2-xx, #3-xx, #5-xx, #6-xx |
| QuEL-1 SE Fujitus-11モデル タイプA機 | `quel1se-fujitsu11-a` | QuEL-1 SE #4-xx                             |
| QuEL-1 SE Fujitus-11モデル タイプB機 | `quel1se-fujitsu11-b` | 同上                                          |
| QuBE OU タイプA機                 | `qube-ou-a`           | QuBE OU #1-xx, #2-xx, #3-xx                 | 
| QuBE OU タイプB機                 | `qube-ou-b`           | 同上                                          | 
| QuBE Riken タイプA機              | `qube-riken-a`        | QuBE Riken #1-xx                            | 
| QuBE Riken タイプB機              | `qube-riken-b`        | 同上                                          | 

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

なお、v0.8.9以降では全ての制御装置でJESD204C準拠のリンクキャリブレーションを行い、さらに、使用中も常時キャリブレーションが行われる（バックグラウンドキャリブレーション）を実施する設定となる。
この設定がリンクアップの成功率を最大化し、かつ、リンクアップ後のCRCエラーの発生を抑えることができる。
しかし、なんらかの理由で従来と同様にJESD204B準拠のリンクアップキャリブレーションを使いたい、あるいは、バックグラウンドキャリブレーションを止めたい場合には、
それぞれ `--use_204b` と `--nouse_bgcal`をオプションとして指定すればよい。

#### 制御装置のループバック経路を使った動作確認
##### quel1_check_all_internal_loopbacks.py
QuEL-1 Type-A/B および QuBE RIKEN Type-A/B の信号入出力の健全性を内部ループバック経路を使用して確認するためのツールである。
対象の装置のIPアドレス(`--ipaddr_wss`)と装置モデル名(`--boxtype`)、クロックマスタのIPアドレス(`--ipaddr_clk`)を指定して使用する。
クロックマスタの指定が必要なのは、波形発生を時刻カウンタをトリガにして行っているからである。
なので、対象装置のシーケンサの動作確認も同時に行うことになる。

###### 制限事項
SIMPLEMULTI_STANDARD のファームウェア以外のファームウェアには対応していない。
というのは、Read-inとMonitor-in の同時使用能力とSIMPLEMULTIのタイムトリガ波形生成の機能との両方が必要だからである。

###### 使用例
たとえば、動作確認対象の装置が QuEL-1 Type-A で IPアドレスが`10.1.0.58`、クロックマスタのIPアドレスが`10.3.0.13`の場合、
次のようなコマンドで全ての入出力ポートの動作確認をできる。
```shell
quel1_check_all_internal_loopbacks --ipaddr_clk 10.3.0.13 --ipaddr_wss 10.1.0.58 --boxtype quel1-a
```

QuEL-1 Type-A は4つの入力ポートを持つので、各入力ポートごとに装置内ループバック経路がある出力ポート群の信号を取得し、そのベースバンド信号の波形を表示する。

![quel1a-loopbacks](./images/quel1_loopbacks.png)

一番上のグラフは、ポート0（read-in) でのキャプチャデータである。
このポートに内部ループバックできるのは、ポート1(read-out)だけである。

横軸はキャプチャ開始からのサンプル数、縦軸はADCの読み値である。
サンプリングレートは500Mspsなので、1サンプルが2nsの時間に対応する。
グラフはそれぞれ、ベースバンド信号の実部(青)と虚部(オレンジ)である。

グラフの表示と共に、コンソールログに詳細な情報が出力されている。
```text
2024-09-17 12:15:49,147 [INFO] quel_ic_config_utils.simple_multibox_framework: number_of_chunks: 1
2024-09-17 12:15:49,147 [INFO] quel_ic_config_utils.simple_multibox_framework:   chunk 0: 63 samples, (470 -- 533),  mean phase = -136.6
```
ここから読み取れるのは、キャプチャ開始後470サンプル目から、長さが63サンプルの矩形波が受信できたことである。
実際には、ポート1から64サンプルの矩形波を出力しているが、パルスの検出閾値如何で2サンプル程度は前後するので、正しくキャプチャできていると言える。
信号発生開始と同時にキャプチャも開始しているが、ADC及びFPGAでの信号処理の遅延などで、波形が得られるまで間がある。
この遅延量はMxFEのリンクアップ毎にバラツキがあるが、再リンクアップするまでは一定に保たれる。
ここで `q`を押すと、次のキャプチャに進む。

2つめのグラフは、ポート5（monitor-in)のキャプチャデータである。
ポート1, ポート2, ポート3, ポート4 の出力をループバックする内部経路があるが、ポート1については既に確認できているので省いている。
したがって、3つのポートから、それぞれ時間差で64サンプルのパルスを出力している。
ポート2の波形はキャプチャ開始と同時に、ポート3の波形は512サンプル遅れ、ポート4の波形は1024サンプル遅れで出力している。
なお、ポート3のSMA端からの出力は2逓倍した信号になるが、ループバックされるのは逓倍前の信号なので観測できている。

以下に示すコンソールログからも、実際の出力波形と整合的な結果が得られていることが分かる。
```text
2024-09-17 12:15:49,280 [INFO] quel_ic_config_utils.simple_multibox_framework: number_of_chunks: 3
2024-09-17 12:15:49,280 [INFO] quel_ic_config_utils.simple_multibox_framework:   chunk 0: 64 samples, (469 -- 533),  mean phase = 86.5
2024-09-17 12:15:49,280 [INFO] quel_ic_config_utils.simple_multibox_framework:   chunk 1: 129 samples, (980 -- 1109),  mean phase = -39.6
2024-09-17 12:15:49,280 [INFO] quel_ic_config_utils.simple_multibox_framework:   chunk 2: 65 samples, (1493 -- 1558),  mean phase = -57.7
```

この後、ポート7(read-in)、ポート12（monitor-in)　と続くが同様のデータが得られる。

###### quel1se_riken8_check_all_internal_loopbacks.py
上記のスクリプトの QuEL-1 SE RIKEN-8 版である。
使い方は基本同じだが、`--boxtype`を与える必要はない。
QuEL-1 SE RIKEN-8 はそれまでのQuEL-1 と異なり、モニタ系の入力にLNAを持たないので、モニタ系でキャプチャした信号の振幅がリード系の 1/10 程度になる。


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

## APIをちらっと見る
以下に 主要APIを項目ごとに列挙する。
詳しい使い方は、`help(box.dump_box)` のように `help`関数で見られたい。

- 全体設定
  - **dump_box**: 制御装置全体の設定状態をデータ構造として得る。
  - **config_box**: 制御装置全体の設定を一括して行う。dump_boxで取得したデータ構造の "ports" の部分と同じデータ構造の全部か一部かを与える。

- ポート設定
  - **dump_port**: 各ポートの設定状態をデータ構造として得る。
  - **config_port**: 指定のポートのパラメタを設定する。

- チャネル設定
  - **dump_channel**: 指定の出力ポートの出力チャネルの設定状態をデータ構造として得る。
  - **config_channel**: 指定の出力ポートの出力チャネルのパラメタを設定する。

- runit設定
  - **dump_runit**: 指定の入力ポートの runit の設定状態をデータ構造として得る。
  - **config_runit**: 指定の入力ポートの runit のパラメタを設定する。

- RFスイッチ
  - **dump_rfswitches**: 全てのポートのRFスイッチの状態を取得する。
  - **config_rfswitches**: 全てのポートのRFスイッチの状態を設定する。
  - **block_all_output_ports**: 全ての出力ポートのRFスイッチをblock状態にする。ただし、モニタアウトは含まない。なお、以下の4つのAPIは頻繁なユースケース向けに config_rfswitches を特化したものである。 
  - **pass_all_output_ports**: 全ての出力ポートのRFスイッチをpass状態にする。ただし、モニタアウトは含まない。
  - **activate_monitor_loop**: モニタ系のRFスイッチをループバック状態にする。
  - **deactivate_monitor_loop**: モニタ系のRFスイッチのループバックを解除する。

- 信号出力API（本文書では説明しない。詳しくは[こちら](./MIGRATION_TO_0_10_X.md)を参照。）
  - **initialize_all_awgunits**: 全てのAWGを初期化する。
  - **initialize_all_capunits**: 全てのキャプチャユニットを初期化する。
  - **get_current_timecounter**: 現在の時刻カウンタを取得する。
  - **register_wavedata**: 指定の出力チャネルに波形データを名前を付けて登録する。波形チャンクを作成するのに使用する。
  - **start_wavegen**: 制御装置内の複数の出力チャネルを同時に起動する。即時または指定時刻カウンタでの起動が可能。
  - **start_capture_now**: 制御装置内の複数の runit を同時に即時起動する。
  - **start_capture_by_awg_trigger**: 制御装置内の複数の出力チャネルと複数の runit を指定時刻に一斉起動する。

### コード例
QuEL-1 SE RIKEN8 モデルの制御装置で、全ての出力ポートから短い方形パルスを出力し、ループバック経路を介して、全ての入力ポートでキャプチャするサンプルを紹介する。
現在は、quel1se-riken8 用のスクリプトしか用意していないが、近いうちに他の機種用のスクリプトも用意する。
quel1se-riken8 が手元にあれば、次のコマンドをIPアドレスを指定して実行して頂きたい。
```bash
python scripts/simple_timed_loopback_example_quel1se_riken8.py --ipaddr_wss 10.1.0.xx
```
すると、次の様な画面が見られるはずだ。

![quel1se-riken8-loopbacks](images/quel1se-riken8_loopbacks.png)

3つのグラフとも、縦軸が信号振幅、横軸がサンプル数である。
一番上のグラフが、ポート0の読み出し信号入力のキャプチャである。
ここには、ポート1の読み出し信号出力から出力した、長さ64サンプルの矩形パルスがループバック経路を介してキャプチャできている。
二番目のグラフは、ポート4のモニタ入力のキャプチャであり、ポート1のFogi信号出力、ポート2のポンプ出力、ポート3の制御信号出力から、それぞれ時間差
で出力した、長さ64サンプルの矩形パルスが得られている。
三番目のグラフは、ポート10のモニタ入力のキャプチャであり、ポート6, 7, 8, 9 の各制御信号出力から、時間差で出力した長さ64サンプルの矩形パルスが得られている。

このスクリプトは、先述の`quel1se_check_all_loopbacks`コマンドと同様の機能を実現するが、コードサンプル向けに平易な書き方をしている。
このコードから、各ポートの設定、波形パラメタの作成方法、キャプチャパラメタの作成方法、波形生成とキャプチャの同時開始、といった、
実際の量子実験用のコード開発を行うための基本的な流れを掴み取ることができると思うので、読解することをお勧めする。

### インタラクティブに実行してみる
上述のコードは、コマンド引数処理のコードを除去してjupyterサーバ上で実行することも可能である。
各部を改変するなどして、各APIの動作を理解する助けとして頂きたい。

また、jupyterサーバ以外にも、インタプリタのインタラクティブシェルを使って、各部の動作確認をするためのサンプルを用意している。
```shell
python -i scripts/getting_started_example.py --boxtype quel1se-riken8 --ipaddr_wss 10.1.0.xxx
```
とすると、指定装置を抽象化したオブジェクトが`box`変数に入った状態で、インタプリタが起動する。
たとえば、`box.dump_box()` とすることで、上述の `quel1_dump_port_config` コマンドの出力同様の結果を含んだデータ構造を得られる。

発展的な例として、先ほどのサンプルスクリプトの各部を切り出し、繋ぎ合わせて、CW信号を出すサンプルを書いてみよう。
ポート3にスペクトラムアナライザをつなぐと、4.0GHzのピークが見られるはずだ。

```python
# 周波数0 のベースバンド波形を登録。
cw_iq = np.zeros(64, dtype=np.complex64)
cw_iq[:] = 32767.0 + 0.0j
# "cw32767" という名前で、ポート3 の チャネル0 に cw_iq の内容を登録する。
# 名前はチャネル内で唯一であれば良い。
box.register_wavedata(3, 0, "cw32767", cw_iq)

# 上記波形を、最大の繰り返し回数出力するように設定する。事実上、待っていても波形生成は終わらない。
ap = AwgParam(num_repeat=0xFFFF_FFFF)
ap.chunks.append(WaveChunk(name_of_wavedata="cw32767", num_blank_word=0, num_repeat=0xFFFF_FFFF))

# 各パラメタをポート3のチャネル0に設定
box.config_port(port = 3, cnco_freq = 4.0e9, fullscale_current=40000, rfswitch="pass")
box.config_channel(3, 0, fnco_freq=0, awg_param=ap)

# 波形生成を即時開始
task = box.start_wavegen({(3, 0)})

# 波形生成を中止する場合は以下のようにする。
task.cancel()
task.result()  # 波形生成をキャンセルしたので、CancelledError の例外が出るのが正常。
```

## 次のステップ
[scriptディレクトリ](../scripts) にある他のスクリプトも実験コードを書く参考になると思う。
これらのスクリプトの内容について、 [こちら](./HOW_TO_USE_EXAMPLES.md)に詳しい説明があるので参照されたい。

また、jupyterlab 上での使用については、[チュートリアル](../../quel_tutorial/README.md) 
が参考になる。
