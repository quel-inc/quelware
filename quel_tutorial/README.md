# quel_tutorial

quel_tutorial は、キュエル社の制御装置を使用して量子実験を行うためのチュートリアルです。
各チュートリアルは Jupyter notebook で構成されています。
その他、各ノートブックを実行するための必要なファイルを含みます。

## 実行環境の構築

チュートリアルを実行するソフトウェア環境を構築します。
[quel_tutorial インストールガイド](https://github.com/quel-inc/quel-tutorial/blob/develop/SETUP.md)の内容を実行することで、本チュートリアルを実行するためのソフトウェア環境を構築することができます。

## 必要となる知識

本チュートリアルを理解するためには、以下のような知識が必要です。

- 二準位系の基本的な物理
- 超伝導量子ビットに関する基本的な知識

量子ビットやそのダイナミクスに関する物理 (Rabi 振動など) については、必要に応じて一般的な書籍を参照してください。
また、本チュートリアルにおいては、超伝導量子ビットの制御をベースに構成しています。
そのため、超伝導量子ビットの基本的な物理や、制御方法および読み出し方法といった基本的な知識があれば、より深い理解につながると思います。
超伝導量子ビットの物理に関しては、[超伝導量子ビットのレビュー論文](https://arxiv.org/abs/1904.06560)などを参照してください。

## コンテンツ

チュートリアルのコンテンツとしては、現時点で QuEL-1 に対応したサンプルコードが利用可能です。
なお、他のモデルにも対応したサンプルコードも順次増やしていく予定です。

### [quel_tutorial for QuEL-1](./quel1)

QuEL-1 の操作方法と QuEL-1 を用いた量子実験がメインの内容です。


## Acknowledgements

We would like to express our sincere appreciation to Prof. Ogawa and Prof. Shiomi at Osaka University for the original implementation and contributions to this project.
