# Software for QuEL's Quantum Computing Control Systems

## quel_ic_config
本リポジトリの中心的パッケージ。
制御装置内のICやRFコンポーネントを設定するための低レベルライブラリを基盤に、各部を抽象化した上で
統合し、量子ビットの制御装置としてのAPI群を定義する。

## reference_manuals
制御装置とその周辺装置のリファレンスマニュアル。

## quel_tutorial
制御装置を量子実験で使う例のノートブック。
制御装置モデルごとに、いくつかのシナリオを公開している。

## e7awghal
制御装置内の波形生成と取得を担うFPGA上のファームウェアを制御するための低レベルライブラリ。
e7awgsw と quel_clock_master が担っていた機能を代替する。

## quel_staging_tool
ファームウェアの配布用パッケージ。

## quel_inst_tool
自動テストでスペクトラムアナライザなどの測定装置を使用するためのライブラリ。

## quel_pyxsdb
ファームウェアアップデートやFPGAのJTAG経由でのリセットなどを支援するためのライブラリ。

## quel_cmod_scripting
QuBE 及び QuEL-1 の恒温制御ファームウェアのシリアルインターフェースにアクセスするためのライブラリ。
