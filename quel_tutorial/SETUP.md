
# quel_tutorial インストールガイド

quel_tutorial のインストールと初期設定の手順について説明します。
なお、推奨する quel_tutorial の実行環境は Ubuntu 20.04 で、OS 付属の Python 3.9 を利用することを想定しています。

## 環境構築

### Visual Studio Code のインストール

本ガイドでは、quel_tutorial の実行に Visual Studio (VS) Code を使用することを想定しています。
以下のコマンドを実行することで、VS Code をインストールできます。
```
sudo snap install code --classic
```
インストール完了後、VS Code を起動し、Jupyter Extension をインストールしておきます。


### ワークスペースの準備

quel_tutorial を実行するためのワークスペースを設定します。
以下のコマンドでディレクトリを作成し、移動します。
```shell
mkdir -p ~/workspace/
cd ~/workspace/
```
このディレクトリを VS Code のワークスペースとして設定します。

### quelware のインストール

quelware を、[公式ドキュメント](https://github.com/quel-inc/quelware/blob/9ec45440ab9bd9f251edb6fefc62c68313441145/quel_ic_config/docs/GETTING_STARTED.md)に従ってインストールします。
インストール後、制御装置のリンクアップを[公式ドキュメント](https://github.com/quel-inc/quelware/blob/9ec45440ab9bd9f251edb6fefc62c68313441145/quel_ic_config/docs/GETTING_STARTED.md#%E3%83%87%E3%83%BC%E3%82%BF%E3%83%AA%E3%83%B3%E3%82%AF%E7%8A%B6%E6%85%8B%E3%81%AE%E7%A2%BA%E8%AA%8D)に示された手順に従って行ってください。

### quel_tutorialのインストール

quel_tutorial のリポジトリをクローンし、必要なセットアップを行います。

```shell
git clone https://github.com/quel-inc/quel_tutorial.git
cd quel_tutorial
```

以上で準備完了です。

