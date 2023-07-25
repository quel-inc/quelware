# Software for QuEL's Quantum Computing Control Systems

## quel_ic_config
制御装置内の各コンポーネントを設定するための低レベルライブラリ。
量子実験用のライブラリに、制御装置のハードウェアの詳細を隠蔽した汎用的にインターフェースを提供する。

## quel_clock_master
制御装置間での時刻同期及びタスクスケジュールに関わるモジュールを操作するためのライブラリ。
[こちら](quel_ic_config/scripts)に使用例のコードがある。

## quel_inst_tool
自動テストでスペクトラムアナライザなどの測定装置を使用するためのライブラリ。
自動テストの依存ライブラリであるので、ここで開発を行う。
