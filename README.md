# Software for QuEL's Quantum Computing Control Systems

## quel_ic_config
本リポジトリの中心的パッケージ。
制御装置内のICやRFコンポーネントを設定するための低レベルライブラリを基盤に、各部を抽象化した上で
統合し、量子ビットの制御装置としてのAPI群を定義する。

## e7awghal
制御装置内の波形生成と取得を担うFPGA上のファームウェアを制御するための低レベルライブラリ。
e7awgswが担っていた機能を代替する。

## quel_clock_master
制御装置間での時刻同期及びタスクスケジュールに関わるモジュールを操作するためのライブラリ。
まもなく e7awghalで置き換える予定。

## quel_inst_tool
自動テストでスペクトラムアナライザなどの測定装置を使用するためのライブラリ。

## quel_pyxsdb
ファームウェアアップデートやFPGAのJTAG経由でのリセットなどを支援するためのライブラリ。

## quel_staging_tool
ファームウェアの配布用パッケージ。
