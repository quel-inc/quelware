# Software for QuEL's Quantum Computing Control Systems

## Getting Started

### Cloning the Repository

To clone this repository, run the following command:

```
git clone git@github.com:quel-inc/quelware-internal.git
```

To work with firmware images, you will need to check out `quel-firmware` directory, which is managed as a Git submodule.
Before doing so, please make sure you have installed [Git LHS](http://git-lfs.com/) on your system.
You can find the official installation instructions [here](https://github.com/git-lfs/git-lfs?utm_source=gitlfs_site&utm_medium=installation_link&utm_campaign=gitlfs#installing).

To initialize and check out the submodules, please run:

```
git submodule update --init --recursive
```


## Packages

### quel_ic_config
本リポジトリの中心的パッケージ。
制御装置内のICやRFコンポーネントを設定するための低レベルライブラリを基盤に、各部を抽象化した上で 統合し、量子ビットの制御装置としてのAPI群を定義する。

### quel_clock_master
制御装置間での時刻同期及びタスクスケジュールに関わるモジュールを操作するためのライブラリ。

### reference_manuals
制御装置とその周辺装置のリファレンスマニュアル。

### quel_inst_tool
自動テストでスペクトラムアナライザなどの測定装置を使用するためのライブラリ。

### quel_pyxsdb
ファームウェアアップデートやFPGAのJTAG経由でのリセットなどを支援するためのライブラリ。

### quel_staging_tool
ファームウェアの配布用パッケージ。

### quel_cmod_scripting
QuBE 及び QuEL-1 の恒温制御ファームウェアのシリアルインターフェースにアクセスするためのライブラリ。
