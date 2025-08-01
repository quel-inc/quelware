# 開発背景・基本設計方針

## 能書き
ユーザが必要とする制御装置の機能拡張をサクッと開発・提供するための枠組みを作る。

### ライブラリ開発の効率化
型ヒントとユニットテストが整備されており、また、関連する実装がコードの字面上で局在するように設計しているので、
機能追加によって予想外のパンチを食らう可能性をかなり低減できている。
従来のように、Cで記述した実行ファイルを呼び出すようなこともないので、異常の検出も容易になっている。

### ハードウェア・ファームウェアの変更への適応
FY2022のハードウェア開発時には、ソフトウェアの変更がまったく間に合わなかった。
FY2023はハードウェア納品時には、基本的な機能を網羅したソフトウェアも提供する。

#### 構成情報のデータ化
ハードウェアのモジュールの差し替えや追加に対して、局所的なコード改変で対応できるような設計になっている。

####  初期設定のコードからの分離
各世代および各タイプの制御装置ごとに、各種ICのレジスタの初期値をコードからなるべく独立して持てるようにした。
製品出荷時に標準の初期設定を付属するが、ユーザはオリジナルの初期値を作成することもできる。この際にコードを修正せずに済むのは
好ましい性質だ。

#### RAII?
コンストラクタで対応するファームウェアへの疎通確認まで済ませるか否かは、迷うところではある。
なんだが、ハードウェアコンポーネントがネットワークの向こう側におり、疎通確認に不確実性があるので、
今回はコンストラクタはハードウェアに触る準備（たとえば、OSが提供するリソースを確保する）までを分担し、
ハードウェアに実際に触りに行くのは、その後にするという設計ポリシーを採った。
このポリシーの嫌なところは、ハードウェアに触って初期化をする必要があるときに、面倒が発生するところにある。
つまり、コンストラクタを呼んだ後、他のメソッドを呼ぶ前に必ず初期化をしなければいけない、のだが、解決策が大なり小なり煩わしい。
一番簡単なのは、ユーザに明示的に初期化メソッドを呼ぶことを要求することだが、これは不親切すぎる。
また、初期化の結果に依存した処理をする各所で、何らかのチェックをする羽目になるのも厄介に拍車を掛ける。
そうなると、公開メソッドの先頭で必ず初期化の有無を確認し、まだであれば、自動的に初期化をする、という話になると思うのだが、
これも、各公開メソッドの先頭にボイラープレートを書かないといけないのが煩わしい。
なんだが、ここでは後者で行くことにする。

## 全体像とか設計とか
cssの開発をしようとして始めたのだが、テスト用に書いたコードを発展させて、より上位のレイヤにまでカバーする方向で発展している。
0.7.4 までは、正式にサポートするのはcssだけであり、box層、あるいは、box内でcssと同じ階層に並ぶwss層は、util扱いであった。
0.8.7 にて、box層は[大幅な改修](MIGRATION_TO_0_8_X.md)の上、正式サポートに格上げとなった。
同時にwss層も正式サポート扱いになっているが、wss層は公開していないので、やや大胆な変更が入る余地を残しておきたい。

### 階層構造
box の下に、 wss / sss / css の各コンポーネントが居る、という設計。
単一の制御装置についてはboxレベルのAPIを叩くことで、好きなことができるようになるのが最終形。

最終的には、box層 の上に 量子コンピュータ制御システムの層を作って複数台を協働させる予定だが、設計中である。
testlibsディレクトリの中に実験的な実装があり、複数台を協働させるサンプルスクリプトと共に公開している。
この実装では、sss を box に取り込んでいないが、近い将来に取り込む予定である。

### css
#### 送信系
従来ライブラリを初めて触ったときの障害のひとつが、各種リソース間のマッピングがグチャグチャしていることだ。
ドキュメントにちょっとした誤記があったり、コードにもちょっとしたバグがあったり、と何が正しいのか分からないので、
実験して確かめる羽目になったりと、開発速度が上がらない原因のひとつになっている。

そこで、パワーユーザ及び開発者向けに、box, group, line, channel という人間にとって比較的理解しやすい概念を導入する。
これらは、2つの種類の概念、電子回路基板やRTLの設計に由来する各種インデクスと制御装置の筐体の外に出ているSMAコネクタ番号（いわゆるポート番号） との橋渡しをする。
注意するべきは、前者の概念が普通のユーザにとって意識する必要性が薄いのに対し、後者はユーザにとって重要度が高いことだ。
たとえば、10個あるLMX2594に振られた番号や、1つのAD9082が持つ4つDACの番号であるが、そういった詳細なインデクスはパワーユーザであっても常に意識する必要はない。
一方で、それぞれのポートからどの種類の信号が入出力可能で、また、それらを実験コードから参照する方法は、ユーザの重大な関心事である。
したがって、これら２つを直接紐付けても、誰もあまり得をしない。

そこで、導入するのが最初に述べた４つの概念である。
- **box**: 制御装置を指定する識別子
- **group**: なんらかのリソースを共有する、ひとかたまりのDAC/ADCのグループ。
  - QuBE-OU, QuBE-RIKEN, QuEL1 では、groupとMxFE (AD9082) が対応していると考えて問題ない。
  - QuEL1-NEC では、groupと制御対象となる量子ビットと対応が取れる。
- **line**: グループ内の信号経路の識別子。1つのDAC出力ピンあるいは1つのADCの入力ピンに対応する。
  - たとえば、QuBE-OU, QuBE-RIKEN, QuEL1 では、AD9082の4つのDACのそれぞれが Readout, Pump, Ctrl, Ctrl の機能を持つが、AD9082内のDACのハードウェアIDと機能のマッピングは単純ではない。line はこのマッピングを分かりやすくするために導入された。
- **channel**: 1つのDAC及びADCが複数個持つ得るチャネルと対応する。ICは全てのチャネルを通し番号で管理するが、ユーザにとっては、各line内で閉じたインデクスで参照する方が便利なので、番号を振り直すのに用いる。この概念は、FPGA内のAWG Unit及びCapture Moduleのインデクスとも対応する。

##### 例：QuEL-1 Type-A の場合
前半ポート（0から6)が グループ0、後半ポート（7から13)が グループ1 に対応している。
各グループは4つのDACライン（0から3）と2つのADCライン（0と1）を持つ。
４つのDACラインは、0がRead-out, 1がPump, 2がひとつめのCtrl, 3がふたつめのCtrlに対応する。
２つのADCラインは、0がRead-in, 1がMonitor-in に対応する。
簡単のために番号で呼んでいるが、適切な名前を付けるともっと便利になるだろう。

| MxFEの番号 | DAC番号 | (group, line)　 | ポート番号 | 機能 |
|-----------|--------|----------------|-------|------|
| 0       | 0     | (0, 0)         | 1     | Read-out |
| 0       | 1     | (0, 1)         | 3     | Pump |
| 0       | 2     | (0, 2)         | 2     | Ctrl |
| 0       | 3     | (0, 3)         | 4     | Ctrl |
| 1       | 3     | (1, 0)         | 8     | Read-out |
| 1       | 2     | (1, 1)         | 10    | Pump |
| 1       | 1     | (1, 2)         | 11    | Ctrl |
| 1       | 0     | (1, 3)         | 9     | Ctrl |

歴史的経緯などで、リソースのマッピングが非常に複雑なことになっている。
これにチャネルとかAWGとかがad hocにアサインされているのが組み合わさるとどうなるか、想像に難くないだろう。
このライブラリはユニットテストで、上記のものを含む全てのリソースマッピングについて、テストコードを走らせてテストする方針なので、
参照文献としても有用なものになるだろう。

##### 例：Quel-1 Type-B の場合
Type-A では、コントロールポート間のクロストークの予防策として、同じグループの2つのコントロールポートが物理的に隣接しないようなポート配置
をしていたが、Type-Bでは、全ての出力ポートがコントロールポートなので、そのような予防策を行う余地がない。
そうであっても Type-A と同じポート配置にしておくという選択はあったのだが、実際には、内部の配線が自然な状態に近くなるようなポート配置になっている。
この差異は、想像以上に頭がこんがらがるので、group, line という概念が必要になった大きい理由のひとつになってる。

なにはともあれ、次の表のような結線になっている。

| MxFEの番号 | DAC番号 | (group, line)　 | ポート番号 | 機能   |
|-----------|--------|----------------|-------|------|
| 0       | 0     | (0, 0)         | 1     | Ctrl |
| 0       | 1     | (0, 1)         | 2     | Ctrl |
| 0       | 2     | (0, 2)         | 3     | Ctrl |
| 0       | 3     | (0, 3)         | 4     | Ctrl |
| 1       | 3     | (1, 0)         | 8     | Ctrl |
| 1       | 2     | (1, 1)         | 9     | Ctrl |
| 1       | 1     | (1, 2)         | 11    | Ctrl |
| 1       | 0     | (1, 3)         | 10    | Ctrl |

##### 例: QuBE の場合
QuBEはQuEL-1の元になった制御装置であり、最初型（QuBE-OU）とその改良版（QuBE-RIKEN）の2つのモデルがある。
表面的には、QuBE-OUにはモニタ系が無く、それにモニタ系を追加したのがQuBE-RIKENである。
それぞれ、2つの構成の装置、Type-AとType-B があり、先述のQuEL-1のType-AとType-Bにも踏襲している。

QuEL-1 とはポートの並びが大きく異なることに注意が必要である。

| MxFEの番号 | DAC番号 | (group, line)　 | ポート番号 | Type-A機能 | Type-B 機能 |
|-----------|--------|----------------|-------|----------|-----------|
| 0       | 0     | (0, 0)         | 0     | Read-out | Ctrl      |
| 0       | 1     | (0, 1)         | 2     | Pump     | Ctrl      |
| 0       | 2     | (0, 2)         | 5     | Ctrl     | Ctrl      |
| 0       | 3     | (0, 3)         | 6     | Ctrl     | Ctrl      |
| 1       | 3     | (1, 0)         | 13    | Read-out | Ctrl      |
| 1       | 2     | (1, 1)         | 11    | Pump     | Ctrl      |
| 1       | 1     | (1, 2)         | 8     | Ctrl     | Ctrl      |
| 1       | 0     | (1, 3)         | 7     | Ctrl     | Ctrl      |

##### 例：Quel-1 NEC の場合
NECカスタム機は、4つの量子ビットのそれぞれに、観測出力、ポンプ、観測入力の3つのポートを用意した機体である。
4つ全てのADCを観測入力として使用する。
なお、量子ビットの多重化読み出しには対応していない。

| MxFEの番号 | DAC番号 | (group, line)　 | ポート番号 | 機能       |
|-----------|-------|----------------|-------|----------|
| 0       | 0     | (0, 0)         | 0     | Read-out |
| 0       | 2     | (0, 1)         | 1     | Pump     |
| 0       | 1     | (1, 0)         | 3     | Read-out |
| 0       | 3     | (1, 1)         | 4     | Pump     |
| 1       | 2     | (2, 0)         | 6     | Read-out |
| 1       | 0     | (2, 1)         | 7     | Pump     |
| 1       | 3     | (3, 0)         | 9     | Read-out |
| 1       | 1     | (3, 1)         | 10    | Pump     |

##### 例：QuEL-1 SE Riken8 の場合
前半ポート (0から5) が グループ0、後半ポート (6から11) が グループ1としているが、
DACないしADCが所属するAD9082の番号をそのままグループと解釈したという以上の意味を持たない。
各グループは4つのDACライン（0から3）と2つのADCライン（0と1）を持つ。
QuEL-1 SEでは、従来の機種と異なり、グループ0とグループ1が対称的ではないことも、グループに意味を持たせづらいことの理由になっている。
強いて言えば、モニタ系を指定するのに便利である。
というのは、モニタ出力としてコンバインされるDACは、同じAD9082ごとにまとめられているからである。

| MxFEの番号 | DAC番号 | (group, line)　 | ポート番号 | 機能 |
|-----------|-------|----------------|-------|--|
| 0       | 0     | (0, 0)         | 1     | Read-out |
| 0       | 1     | (0, 1)         | 1     | Fogi |
| 0       | 2     | (0, 2)         | 2     | Pump |
| 0       | 3     | (0, 3)         | 3     | Ctrl |
| 1       | 0     | (1, 0)         | 6     | Ctrl |
| 1       | 1     | (1, 1)         | 7    | Ctrl |
| 1       | 2     | (1, 2)         | 8    | Ctrl |
| 1       | 3     | (1, 3)         | 9     | Ctrl |

##### 今後の方向性
quel_ic_config はAD9082の設定変更を積極的に行える枠組みも提供する。
なので、上記の歴史を引きずった不規則なリソースアサインの整理を安全に行うことができるようになるだろう。
すでにリリース済みの機種については、変更をしないつもりだが、今後の機種についてはリリース前にリソースマッピングを整理していく。

#### 受信系
受信系のリソースも正確に理解するのが難しい。
ハードウェア及びファームウェアでのリソースの考え方が発展途上にあるので、将来の拡張を妨げない概念形成が求められていること相まって、事態は複雑である。
いずれにしても、送信系と同様に、パワーユーザ向けに次のような階層的概念を導入する。
box, group までは共通なので、それよりも細かい概念を説明する。

- **rline**: アナログ信号経路の識別子。つまり、ADCのポートに紐付けられる。
- **runit**: rline内に複数あるいくつかの信号処理経路の識別子。同じ　rline 内の runit は同じアナログをADCしたデータストリームを受け取るが、途中のフィルタ、ダウンコンバータ、各種DSPの設定に依存して異なる出力を生成する。

量子計算機の言葉で言えば、rline は基本的に同じ読み出し信号によって反射測定される量子ビットクラスタと対応し、runit がそのクラスタ内の各量子ビットに対応する。

これに加えて、`rchannel` という概念が存在する。
これの理解はハードウェア構成丸出しになってしまい、少々厄介なのだが、現状では意識する必要が薄い。
`rchannel` は、AD9082のFDUCの抽象化でありFNCO周波数の設定に関わるが、現状では1つの`rline`の全ての`runit`が同じ`rchannel`
に属しているので、`rline`と区別する必要性が薄い。
また、FNCO周波数は0で運用するのが普通なことも、`rchannel`の存在を目立たなくしている。
将来的に、ひとつの入力ポートの受信帯域を広げる目的で `rchannel` が前面に出てくる可能性がある。
ただし、`rchannel`を`runit`の属性として表現することで、APIへの影響は最小限に抑えられると考える。

##### rlineについて
各グループは、リード系(`r`)とモニタ系(`m`)の２つの rline を持ち得る。
設計意図としては、リード系が量子ビットの反射測定用の系で、モニタ系が装置自体が生成した信号をループバックで監視するための系である。
これら2つのrlineは、FPGAの手前までは独立したハードウェアリソースによって実現されているが、2024年1月25日より前の標準ファームウェア(simplemulti) では、
FPGAの入り口でどちらか一方が選択されて、共通のキャプチャモジュールへ接続する。

2024年1月25日以降のファームウェアでは、リード系が現状のキャプチャモジュールを専有し、モニタ系用に簡易版のキャプチャモジュールを新設している。
この形態が今後のスタンダードになるので、 基本的にひとつのrlineはひとつのキャプチャモジュールに紐づく、と考えてよい。
逆が成り立たないことに気をつければ、この言明は従来のファームウェアでも正しい。

##### runitの実現について
runit は QuEL-1のAD9082のADCのチャネライザとFPGA内のDSPモジュールによって実現される機能を抽象化した概念である。
runitとDSPモジュールの間には一対一の対応関係が成り立つが、runitとチャネライザとの間は必ずしも成り立たない。
例えば、現状の simplemulti ファームウェアでは、同じ rline内の全てのrunit（あるいは「同じcapture module内のcapture unit」と言い換えてもよい）
が、ひとつのチャネライザを共有している。

この単一チャネライザ構成は、共通の読み出し信号でカバーされる各量子ビットの読み出し共振器の共振周波数が、ひとつのチャネライザの帯域内に収まっている前提の設計である。
この制限を緩和したい場合には、複数のチャネライザを割り当ればよいが、単一チャネライザ構成とrunitとチャネライザを一対一とする構成との間に、いくつかの中間的な構成が可能である。
可能性な概念化として、rline - チャネライザ - runit という3層構造があるが、これは不採用とした。
つまり、runit は個々の割り付け方式と独立した概念とし、runitとチャネライザとの対応関係はrunitの属性として管理する方が、使いやすいと考えた。
実際には「中間的な割り付け」は可能な限り避けることになるはずで、そのような可能性のためにリソースの概念を複雑化するのを嫌った、ということでもある。

この発想は、将来的にユーザにrunitとチャネライザの紐づけを暗記することを迫るかもしれない。
しかし、この問題はBox層のAPI設計によって解決可能である。
現状でも、実は、出力ポートと入力ポートでのLOの共有や、出力ポートと入力ポートでのスイッチの共有、といったことを暗記する必要がある。
これは、装置を触りはじめたばかりのユーザにとっては、脅威とすら言えるだろう。

このような問題を解決する最も簡単な方法は、 装置全体の設定を一度に行い、直後に設定値の全てが正しく実際の設定に反映されていることを確認すれば良い。
もし、与えた設定値同士で矛盾があれば、設定値が正しく反映されていない箇所が必ず発生する。
Box層のAPIには、任意の粒度での一括設定と確認を自動で行い、最終的な設定状態に齟齬があればユーザに警告を発するものがある。
これをうまく使えば、ユーザの意図に反した設定状態で実験が行われることを未然に防げる。
また、設定が意図通りの状態になっていることを確認するAPIもあるので、設定後の状態が意図通りであることを簡単に確認することができる。
このAPIは様々な使い方できるはずで、たとえば、実験中に不慮の事故などで設定が変わっていないことを実験終了後に確認するのに便利である。

##### 2024年1月25日版より前のsimplemultiファームウェア
QuBE時代からつづく従来のsimplemultiファームウェアでは、読み出し系とモニタ系を同時に使うことができなかった。
リンクアップ時にどちらか一方を選択するので、リード系とモニタ系との両方を同じキャプチャユニットに割り当てている。

| ポート番号 | 受信LO番号 | MxFEの番号 | ADC番号, CNCO番号, FNCO番号 | (group, rline, runit)　 | キャプチャモジュール | キャプチャユニット |
|----|---|---|---------|------------------------|----|---|
| 0  | 0 | 0 | 3, 3, 5 | (0, r, 0)              | 1  | 4 |
| 0  | 0 | 0 | 3, 3, 5 | (0, r, 1)              | 1  | 5 |
| 0  | 0 | 0 | 3, 3, 5 | (0, r, 2)              | 1  | 6 |
| 0  | 0 | 0 | 3, 3, 5 | (0, r, 3)              | 1  | 7 |
| 5  | 1 | 0 | 2, 2, 4 | (0, m, 0)              | 1  | 4 |
| 5  | 1 | 0 | 2, 2, 4 | (0, m, 1)              | 1  | 5 |
| 5  | 1 | 0 | 2, 2, 4 | (0, m, 2)              | 1  | 6 |
| 5  | 1 | 0 | 2, 2, 4 | (0, m, 3)              | 1  | 7 |
| 7  | 7 | 1 | 3, 3, 5 | (0, r, 0)              | 0  | 0 |
| 7  | 7 | 1 | 3, 3, 5 | (0, r, 1)              | 0  | 1 |
| 7  | 7 | 1 | 3, 3, 5 | (0, r, 2)              | 0  | 2 |
| 7  | 7 | 1 | 3, 3, 5 | (0, r, 3)              | 0  | 3 |
| 12 | 6 | 1 | 2, 2, 4 | (0, m, 0)              | 0  | 0 |
| 12 | 6 | 1 | 2, 2, 4 | (0, m, 1)              | 0  | 1 |
| 12 | 6 | 1 | 2, 2, 4 | (0, m, 2)              | 0  | 2 |
| 12 | 6 | 1 | 2, 2, 4 | (0, m, 3)              | 0  | 3 |

##### feedbackファームウェア / 2024年1月25日版以降のsimplemultiファームウェアの場合
feedback版のファームウェア及び2024年1月25日版以降のsimplemulti版のファームウェアでは、リード系とモニタ系を同時に使用できる。
モニタ系用に2つの新たなキャプチャモジュールを追加している。
新しいキャプチャモジュールはDSP機能を省いた簡易版のキャプチャユニットを1つだけ持つ。
つまり、各出力ポートのモニタという生波形をキャプチャするだけの用途に特化した構成になっている。

| ポート番号 | 受信LO番号 | MxFEの番号 | ADC番号, CNCO番号, FNCO番号 | (group, rline, runit)　 | キャプチャモジュール | キャプチャユニット |
|----|---|---------|----------|------------------------|----|----|
| 0  | 0 | 0       | 3, 3, 5  | (0, r, 0)              | 1  | 4  |
| 0  | 0 | 0       | 3, 3, 5  | (0, r, 1)              | 1  | 5  |
| 0  | 0 | 0       | 3, 3, 5  | (0, r, 2)              | 1  | 6  |
| 0  | 0 | 0       | 3, 3, 5  | (0, r, 3)              | 1  | 7  |
| 5  | 1 | 0       | 2, 2, 4  | (0, m, 0)              | 3  | 9  |
| 7  | 7 | 1       | 3, 3, 5  | (1, r, 0)              | 0  | 0  |
| 7  | 7 | 1       | 3, 3, 5  | (1, r, 1)              | 0  | 1  |
| 7  | 7 | 1       | 3, 3, 5  | (1, r, 2)              | 0  | 2  |
| 7  | 7 | 1       | 3, 3, 5  | (1, r, 3)              | 0  | 3  |
| 12 | 6 | 1       | 2, 2, 4  | (1, m, 0)              | 2  | 8  |

##### QuEL-1 NEC の場合
4つ全てのADCを観測入力として用いているのが、この機体の特徴のひとつである。
公式には、それぞれのADCがキャプチャユニットをひとつだけ持つ。
なお、このモデルは、2024年1月25日より前のsimplemulti版ファームウェアをサポートしていない。

| ポート番号 | 受信LO番号 | MxFEの番号 | ADC番号, CNCO番号, FNCO番号 | (group, rline, runit)　 | キャプチャモジュール | キャプチャユニット |
|-------|--------|---------|-----------------------|------------------------|------------|-----------|
| 2     | 0      | 0       | 3, 3, 5               | (0, r, 0)              | 1          | 4         |
| 5     | 1      | 0       | 2, 2, 4               | (1, r, 0)              | 1          | 9         |
| 8     | 6      | 1       | 3, 3, 5               | (2, r, 0)              | 0          | 0         |
| 11    | 7      | 1       | 2, 2, 4               | (3, r, 0)              | 0          | 8         |


##### QuEL-1 SE RIKEN-8 の場合
QuEL-1 SE RIKEN-8 モデルは、従来のQuEL-1 Type-A/Bと異なり、1台で4つの量子ビットを制御する設計である。
制御対象の4つの量子ビットは、4多重読み出しできるグループを構成しているので、観測入出力を1組だけ持っている。
モニタ系を2系統もっており、それぞれがひとつのAD9082の4つのDAC出力をモニタする設計である。
受信用のLOを2つのモニタ系で共有していることに注意が必要である。
なお、このモデルは、2024年1月25日より前のsimplemulti版ファームウェアをサポートしていない。

| ポート番号 | 受信LO番号 | MxFEの番号 | ADC番号, CNCO番号, FNCO番号 | (group, rline, runit)　 | キャプチャモジュール | キャプチャユニット |
|-------|--------|---------|----------|------------------------|----|----|
| 0     | 2      | 0       | 3, 3, 5  | (0, r, 0)              | 1  | 4  |
| 0     | 2      | 0       | 3, 3, 5  | (0, r, 1)              | 1  | 5  |
| 0     | 2      | 0       | 3, 3, 5  | (0, r, 2)              | 1  | 6  |
| 0     | 2      | 0       | 3, 3, 5  | (0, r, 3)              | 1  | 7  |
| 4     | 4      | 0       | 2, 2, 4  | (0, m, 0)              | 3  | 9  |
| 10    | 4      | 1       | 2, 2, 4  | (1, m, 0)              | 2  | 8  |

### wss
この API は、AWG Unit / Capture Mod / Capture Unit といった概念を参照する。
quelware-0.10.1で、新規実装のhal層である e7awghal の上に構築した、新しい実装に挿し替えた。
e7awghal は従来の e7awgsw の下半分を置き換えるライブラリであり、リソース管理機能の強化、上位層との連携の整合性の向上、実行速度の最適化、
エラーハンドリングの適正化による安定性の向上など、大幅な改善を実現している。
新しいwss は、e7awgswの上半分の機能を引き受けた上で、複数AWGユニットと複数キャプチャユニットを時刻指定で同時起動するような粒度の高機能APIを
提供しており、さらに、キャプチャデータを構造化してユーザに提供する。

boxレイヤは、wss層のAPIを port, channel, runit といったbox層の概念に紐づける。
で実現している。 
正確には、boxレイヤの概念をcssレイヤの概念に変換した上で、cssレイヤの概念とwssレイヤの概念を対応付けている。
この対応づけの中核は ResourceMapper (rmap) モジュールでおこなっており、 css側のインターフェース mxfe_idx, fduc_idx, fddc_idx、 
wss側のインターフェースが awgunit_idx, capmod_idx となっている。
実は、css側のインターフェースは、css層ではなくて、その下のhal層の概念を用いており、rmapには回路図の情報がそのまま記載されている
と考えて良い。
つまり、rmap は基板構成の変更やFPGAのピンアサインの変更を吸収するための概念装置である。

### sss
wss層に吸収してしまったので、実体としては wss層の一部となっている。


## 制御装置などのアドレスについて
コマンドライン引数のバリデータでは ip_address.IPv4Address でチェックをしているが、内部的にはアドレスを str で持っている。
一般的な使用状況では名前で引いて使う可能性などがあるので、IPv4Address型に制限してしまうのは現時点では乱暴だと感じるからである。
一方で、手動テスト時にIPアドレス直打ちで使うことが多いので、引数のバリデータ的で変なアドレスを弾くのは便利だと思う。
ここらへんも、アドレスの持ち方のポリシーを整備してから、実装をまともにしていく予定である。

なお、UDP/IPをハードウェア的に実現している部分があるので、近い将来にIPv6アドレスが使えるようになる、みたいなことは無いはず。

## 拡張モジュールのStubの管理
### 更新方法
`helper/rebuild_v170.sh` を実行すればよい。
ただし、スタブを生成する前に対象のライブラリをインストールしておくと話が簡単になるので、依存パッケージを所定の場所に配置しておく
必要がある。
その依存パッケージの配置手順については、[`GETTING_STARTED.md`](./GETTING_STARTED.md) に記載がある。
