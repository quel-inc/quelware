# Software for QuEL's Quantum Computing Control Systems

## quel_ic_config
本リポジトリの中心的パッケージ。
制御装置内のICやRFコンポーネントを設定するための低レベルライブラリを基盤に、各部を抽象化した上で 統合し、量子ビットの制御装置としてのAPI群を定義する。

## quel_clock_master
制御装置間での時刻同期及びタスクスケジュールに関わるモジュールを操作するためのライブラリ。

## quel_inst_tool
自動テストでスペクトラムアナライザなどの測定装置を使用するためのライブラリ。

## quel_pyxsdb
ファームウェアアップデートやFPGAのJTAG経由でのリセットなどを支援するためのライブラリ。

## quel_staging_tool
ファームウェアの配布用パッケージ。

## quel_cmod_scripting
QuBE 及び QuEL-1 の恒温制御ファームウェアのシリアルインターフェースにアクセスするためのライブラリ。
