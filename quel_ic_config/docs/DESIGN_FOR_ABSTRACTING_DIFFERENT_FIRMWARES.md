# 構成識別について

## ハードウェア識別
ハードウェア識別子（`Quel1BoxType`) で管理している。
現在は、識別子を装置から自動で得る手段がないので、外部から与える方式を取っている。
コマンドラインでは、`--boxtype` 引数を介して与えるのが標準である。

QuEL-1 SEでは、ExStickGEのファームウェアの柔軟性が大幅に増したので、このファームウェアへ構成情報を問い合わせることが可能になる。
しかし、QuEL-1 classic と QuEL-1 SE の識別は必要なので、QuEL-1 SEであることを同じ方法で指定することになるだろう。
ここで、従来の `Quel1BoxType` に基づいた仕組みを使い続けたいので、QuEL-1 SE では、与えられた`Quel1BoxType`を自動的に細分化して書き換える方法を取る。 
つまり、ー旦、Quel1seGeneric 的な BoxType にしておき、ファームウェアから構成情報を得て、最終的な BoxType になるという流れである。
これをするためには、ファームウェアから構成情報を得るための仕組みを、ExstickgeProxy とは別に持つ必要が生じる。
このデザインは格好が悪いのだが許容する予定だ。
おそらく、create_box_objects() 相当の箇所で、BoxTypeの細分化を行う設計しておけば、そんなに酷いことにならないと思う。
この場合には、ConfigSubsystem とか、ExstickgeProxy の設計には影響がないので、これらのコンポーネントを複雑化しないで済む。

## ファームウェア識別
ファームウェア識別には、ファームウェアが10.1.0.xx 経由で返答するバージョン識別子を利用する。
バージョン識別子は、AWG, CAP, SEQ の各モジュールが持っているが、SEQのバージョン識別子は使用しない。
理由は次のとおり。
- SEQモジュールを持つのは一部のファームウェア(feedback版)に限られる。
- SEQモジュールが存在するファームウェアでも、識別子取得に先立って、専用の初期化手順を踏む必要がある。

現状では、ファームウェアを次の4種類に分類し、その分類情報でその後の処理を調整する。

| 機能分類 (`E7FwType`)    | 概要                               | ADC構成       | SEQ構成 |
|----------------------|----------------------------------|-------------|-------|
| SIMPLEMULTI_CLASSIC  | いわゆる従来ファーム　                      | 切替          | 簡易トリガ |
| SIMPLEMULTI_STANDARD | NEC機及びQuEL-1 SEの標準ファーム           | 同時          | 簡易トリガ |
| FEEDBACK_VERY_EARLY  | feedback版ファームの試作版 <br>（廃止予定)     | 同時 <br>(変則) | 最初期型  |
| FEEDBACK_EARLY       | feedback版ファーム                　　　 | 同時          | 最初期型  |

ファームウエア機能分類は `E7FwType` によって表現している。
[e7workaround.py](quel_ic_config/e7workaround.py) にある`_VERSION_TO_FWTYPE` にて、ファームウェアバージョン識別子
からファームウェア分類へのマッピングを管理している。
2024年の1月後半から、バージョンの最初のアルファベットに意味を与える運用が始まったが、少なくとも当面はこれに依存しないで、表の管理を続けて
行く予定である。
というのは、そのアルファベットだけでは、機能分類に十分な情報が得られない、あるいは、将来得られなくなる可能性が高いので。

### Quel1Feature
ファームウェア識別を直接使って、各種ICの設定値を選択することも不可能ではないが、実際にはファームウェア識別子を、それらの共通機能要素の組み合わせに還元した方が設定を簡素に記述できる。
この共通機能要素を表現するのが `Quel1Feature` である。
現状、`create_box_object()` の中で還元作業をしており、E7FwTypeから、ADC構成に対応した `Set[Quel1Feature]` を作成して ConfigSubsystem のコンストラクタに渡している。


## 設定オプション
ハードウェアの初期化において、「ハードウェア識別子」及び「ファームウェア機能分類」と共に参照する情報に「設定オプション」がある。
リンクアップ時にユーザが指定することで、初期設定情報を切り替えられるのに用いており、ハードウェア
特に、SIMPLEMULTI_CLASSIC のファームウェアの使用時に、ADCの接続元をリンクアップ時に切り替えるのに用いることになる。

リンクアップをしない場合でも、一部の設定を参照する必要があるのだが、
リンクアップしたときに使用した設定オプションと、装置使用時のオプションが食い違っていると、良くないことが起こる可能性を否定できない。
ハードウェア識別子を食い違って与えられてしまうとどうしようもないのだが、設定オプションは直接参照しないで、各種ICのレジスタ設定をチェックすることにして、
食い違いが発生しても問題なく動作するように心がけている。
基本、この方針は維持するが、 ExStickgeにリンクアップ時のオプション指定を記憶させておき、後から参照できるようにするのが正しいように思った。


## 補足：e7awgsw ライブラリとファームウェアとの互換性チェック
[この文書](../docs_internal/BRANCHED_FIRMWARE_PROBLEM.md)に書いてあるとおり、e7awgswライブラリが2つのブランチに分かれてしまっており、
それぞれのブランチが、SIMPLEMULTI_*系のファームウェアとFEEDBACK_*系のファームウェアのそれぞれに対応する。

ライブラリの由来は、[e7workaround.py](quel_ic_config/e7workaround.py) で定義する `detect_branch_of_library()` が決定し、
`E7LibBranch`型の値として表現する。
ライブラリが定義する CaptureModuleの数とSequencerクラスの有無で判定する。
これについては、SIMPLEMULTI_STANDARDへの対応で、いろいろと良くない問題が発生しているので、近い将来にどうするかは検討中である。

このライブラリの由来情報とファームウェア識別子との対応関係が、WaveSubsystem の中の `VALID_HWTYPES_FOR_LIBRARY_BRANCH` の中で管理されている。
なお、WaveSubsystem内では、外の世界でFWと呼んでいるものをHWと呼んでいる。
これは修正してしまい気もしているので、どこかのタイミングでやるかもしれない。