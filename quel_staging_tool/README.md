# ファームウェア書き込みなどのためのツール群

QuEL-1 を及びその後継機のファームウェア書き込みを誰でもできるようにするのが目的です。
使いようによっては様々事故を引き起こし得るので、細心の注意を払ってご使用くださいませ。

## インストール
```bash
python -m build
pip install dist/quel_staging_tool-x.y.z-py3-none-any.whl
```

## 準備
以下のコマンドを実行するシェルでは、次のコマンドを実行して、Vivadoのパスを通しておく必要がある。
```
source /tools/Xilinx/Vivado/2020.1/settings64.sh
```

## AlveoU50
### ファームウェア過去込み
`program_au50` コマンドを使用する。
以下の例は、アダプタID 500202A50xxxA に繋がっているFPGA(Alveo U50)に、simplemulti_20230820 のバージョンの
ファームウェアを書き込む。この際に、MACアドレスを`00-0a-35-16-66-48`、IPアドレスを`10.1.0.250` に設定する。

```bash
quel_program_au50 --adapter 500202A50xxxA --macaddr 00-0a-35-16-66-48 --ipaddr 10.1.0.250 --firmware simplemulti_20230820
```

- アダプタIDは、アダプタに貼ってあるシールに記載されている。xsct コマンド
- MACアドレスは、Alveo U50 に貼ってあるシールに記載のものを使用する。
- IPアドレスは、MACアドレスと紐づけてた値を社内で管理しているので、それを利用する。
- ファームウェアバージョンは、このパッケージに含まれているものから選ぶ必要がある。 `--help` オプションで可能な選択肢を表示できる。新たなバージョンのファームウェアのパッケージへの追加については、別の文書で説明する予定。

指定のMACアドレスとIPアドレスとは、AのケーブルのI/Fに反映される。
QuEL-1 は BのケーブルのI/Fも有効化されるが、そのMACアドレスとIPアドレスは次のルールで決定する。
- MACアドレスは 1LSB足す。ただし、OUI（上位24bit）に桁上りが発生した場合には、エラーが発生する。
- IPアドレスは、第２オクテットに1を足す。上記例であれば、10.2.0.250 となる。


### リブート
アダプタがPCのUSBコネクタに繋がっていると、FPGAが起動しない場合がある。
その場合には、アダプタIDを指定して次のコマンドを実行することで、起動することができる。

```bash
quel_reboot_fpga --adapter 500202A50xxxA
```

このコマンドを起動すると、アダプタの電源がオフになるまでは、QuEL-1本体のパワーサイクルをした際にFPGAが起動するようになる。
アダプタの電源は、USBケーブル経由で供給されるので、USBケーブルを抜かない限りはこのコマンドを実行しなくても、FPGAは何もしなくても起動する。
実際には、USBケーブルを抜いても、ある程度の時間はアダプタ内に情報が残っているようだ。

また、FPGAがフリーズした場合には、このコマンドで復帰できることが多い。

## ExStickGE
### ファームウェア書き込み
`quel_program_exstickge` コマンドを使用する。使い方は、AlveoU50 と同じ。

### リブート
`quel_reboot_fpga` コマンドを使用できる。