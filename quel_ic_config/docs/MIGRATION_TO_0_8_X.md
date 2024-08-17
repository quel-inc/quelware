# ic_config-0.7.x から ic_config-0.8.x への移行について
quel_ic_configパッケージは、Hardware Abstraction (HAL)層の上にConfig Subsystem (CSS)層、さらにその上にBox層、さらにその上に Virtual Port層という階層構造を成している。
従来の0.7.xでは CSS層より上は試作段階であったが、0.8.xではBox層までが実用的段階に漕ぎつけた。
ただし、その過程でBox層に大幅な改訂を行わざるを得なかった。
以降では、改訂を跨いだ旧APIと新APIの連続性を解説し、旧APIを使用したユーザコードのスムーズな移行に役立つであろう情報を提供する。

なお、CSS層と並列に、Wave Subsystem (WSS)層とSync Subsystem (SSS)層があり、Box層がそれらを取りまとめている。
WSS層はとりあえずの実装があり、基本的な一通りの機能を提供しているが、機能拡張に際して抜本的改訂を行う予定がある。
現状の設計では機能拡張で複雑化するAPIの抽象化が十分に行えず、各所に余計な負担を強いることになることが改訂の理由である。
とはいえ、Box層が提供する機能は拡張されこそすれ、既存機能の変更・廃止はあまりないと考える。
したがって、WSS層の改訂に伴うBox層のAPIの変更は軽微なもので済むはずだ。
SSS層はQuel1Box内には影も形もないが、本文書の最後で述べる VirtualPort層のサンプル実装に片鱗を見ることができる。
現状、装置間同期およびタイムトリガ実行の機能を提供するオブジェクトをBoxオブジェクトの外に並置している。
これを設計見直しをした上で、SSS層としてBoxに取り込む予定である。

## HAL層
- ad9082_v106.check_link_status() が廃止。
   - 同名のメソッドをCSS層に新設し置き換え。

## CSS層
- 先述したとおり、check_link_status() メソッドを新設し、AD9082のリンク状態の確認をCSS層でできるようにした。
- dump_config() の出力形式を変更した。これは、後述のとおりBox層からの使用の利便性向上を目的としている。

## Box層
### オブジェクトの生成と初期化
- SimpleBox --> Quel1Box に改名。
- Boxオブジェクトの生成方法が変更。create_box_object() 関数の使用は非推奨。
  - クラスメソッド Quel1Box.create() で代替。
- relinkup()関数、及び reconnect()関数も、それぞれ、relinkup_dev()関数 及び reconnect_dev()関数に改名の上、使用非推奨となった。現状はテストケース内に一部残っているが、いずれ置き換えて廃止となる。
  - Quel1Boxの relinkup()メソッド、あるいは、reconnect()メソッドで、それぞれ代替。
  - なお、SimpleBoxのinit()メソッドは、Quel1Boxには受け継がれていない。上述の relinkup()及びreconnect()メソッドとして再編成した。
- init_box_with_linkup() 及び init_box_with_reconnet() も使用非推奨。
   - create()クラスメソッド 及び、relinkup()メソッドあるいはreconnect()メソッドで置き換え。 

#### 具体例
[getting_started_example.py](../scripts/getting_started_example.py) の変遷を追うと理解しやすい。
0.7.x では、Boxオブジェクトの作成は次のような手順を踏んでいた。
```python
from quel_ic_config_utils import create_box_objects

css, wss, rmap, linkupper, box = create_box_objects(
    ipaddr_wss=str(args.ipaddr_wss),
    ipaddr_sss=str(args.ipaddr_sss),
    ipaddr_css=str(args.ipaddr_css),
    boxtype=args.boxtype,
    config_root=args.config_root,
    config_options=args.config_options,
)

status = box.init(
    ignore_crc_error_of_mxfe=args.ignore_crc_error_of_mxfe,
    ignore_extraordinary_converter_select_of_mxfe=args.ignore_extraordinary_converter_select_of_mxfe,
)
for mxfe_idx, s in status.items():
    if not s:
        logger.error(f"be aware that mxfe-#{mxfe_idx} is not linked-up properly")
```

これが、0.8.x では次のように変わった。
```python
from quel_ic_config import Quel1Box

box = Quel1Box.create(
    ipaddr_wss=str(args.ipaddr_wss),
    ipaddr_sss=str(args.ipaddr_sss),
    ipaddr_css=str(args.ipaddr_css),
    boxtype=args.boxtype,
    config_root=args.config_root,
    config_options=args.config_options,
    ignore_crc_error_of_mxfe=args.ignore_crc_error_of_mxfe,
    ignore_access_failure_of_adrf6780=args.ignore_access_failure_of_adrf6780,
)

status = box.reconnect()
for mxfe_idx, s in status.items():
    if not s:
        logger.error(f"be aware that mxfe-#{mxfe_idx} is not linked-up properly")

```

まず、SimpleBoxはquel_ic_config_utilsパッケージに置いていたが、Quel1Boxはquel_ic_configパッケージに移籍したことに注意が必要である。
Quel1Box.create() は Quel1Boxのオブジェクトだけを返すが、css, wss, rmap linkupper の全てが Quel1Boxオブジェクトに含まれているので何も失っていない。
ただし、これらのオブジェクトはデバグ以外の用途で触る必要はあまり無いと思う。
唯一の例外が、quel_ic_configが未対応の機能を使うために e7awgswを直に叩きたい場合である。
その場合には rmap及びwssを介して、他の部分と整合的にe7awgswの生APIを使うことが可能だ。
当然ながら、この手法の将来に渡る互換性は保証できない。
とはいえ、背に腹は変えられないこともあると思うので、quel_ic_configが将来にサポートした場合の置き換えに備えておきさえすれば、当面の解決策としては妥当であると思う。
なお、create_box_objects() は、box層の実装をするまでもないバラック組みの基板群を動作させるためだけに今後も存続する。
いうまでもなく、装置開発の用途以外での使用は非推奨である。

### コンポーネントの設定
#### 各コンポーネントの設定API
設定と状態確認との各APIを、config_*()系のメソッド群とdump_*()系のメソッド群とに、それぞれまとめた。

まず、装置内コンポーネント概念の階層構造を反映するべく、従来のdump_config()メソッドを dump_box()メソッドに改名した。
box内の各コンポーネントである port, channel, runit　のそれぞれについて、同様のdump_*() メソッドを用意した。
このコンポーネント概念の詳細については、[README.md](../README.md)を参照していただきたい。
これと並置する形で、dump_*()の返り値と同じ形式のデータ構造を引数にとって、ハードウェアの設定に反映する config_*()系のメソッド群を新設した。

- Boxの階層構造に基づいたAPI 
  - dump_box() / dump_port()
  - config_box() / config_port() / config_channel(), config_runit()

config_channel()はDACのFDUCの周波数設定、config_runit()はADCのFDDCの周波数設定を行うが、上位のconfig_port()に含まれているので、
実際に単体で使うことのメリットは薄く、その使用は稀だろう。
また、後述するように config_port() をポートごとに繰り返し使うよりは、confix_box()で一括設定する方が安全である。

#### RFスイッチに特化したAPI
上述のconfig_box()やconfig_port()でも各ポートに付いているRFスイッチを操作することはできる。
しかし、RFスイッチは他のコンポーネントと使用タイミングや使用法がやや異なるケースが多いので、専用のdump/configメソッドに加え、いくつかの特別なメソッドを用意して利便性を図っている。

まず、汎用的な部分については、RFスイッチ専用の dump/configメソッド群でカバーする。
従来の open_rfswitch() メソッドと close_rfswitch() メソッドは廃止とし、config_rfswitch() メソッドで代替する。
さらに、スイッチ単位の状態確認、任意の複数スイッチの一括設定および一括状態確認のためのメソッド群を新設した。
  - dump_rfswitches() / dump_rfswitch()
  - config_rfswitches() / config_rfswitch()

よく使うであろうケースに対応するために、全ての出力ポートを閉鎖・開放するメソッドを新設した。
  - block_all_output_port(), pass_all_output_port()

モニタ系のループの開け締め・状態確認は従来どおり、deactivate_monitor_loop(), activate_monitor_loop(), is_loopback_monitor() のままである。
ただし、リード系についての同様のメソッド群は廃止とし、上述のdump_rfswitch(), config_rfswitch()で代替する。
これは、read_loopのスイッチ状態が、Read-outポートの出力スイッチと連動しているので可能な限り一括設定を使うべき、という意図を反映している。

#### 一括設定の利点
config_box()などの一括設定のメソッドは、設定内容の一貫性確認を自動で行う。
たとえば、Read-inポートと対応するRead-outポートは同じLOを共有しているので、それぞれに異なる周波数の設定することはできない。
ユーザがそのことを忘れていて、それぞれの周波数に違う値を設定してしまった場合、最終状態での実際の設定はユーザの意図と異なるものになってしまう。
これを自動で検知し例外として通知するのが、一貫性確認の機能である。

各項目を個別に設定すると、この安全機能を利用できないので、なるべく一括設定を用いるべきである。
たとえば、ごく一部の設定だけを変更する場合でも、dump_box()で現在の状態を取得して、そこに必要な設定変更を上書きした上で、config_box()で書き戻すという運用によって、
更新設定が意図しない変更を発生していないことを確認できる。
ただし、同じ設定値を再設定した際に、ハードウェアの付随する内部状態が変わってしまう箇所があるので、注意が必要である。
端的には、オシレータの周波数設定を同じ値で上書きすることで、周波数は変わらないが位相が変わってしまう。

実は、この問題には解決策があるので活用していただきたい。
手順としては、次のようになる。
- 現状の設定状態を dump_box() で取得。
- 更新部分だけを config_box() に適用。（自動の一貫性確認は、更新部分だけを確認。）
- 設定前に取得した設定状態のデータ構造に対し、更新部分を上書き。
- config_validate_box() に上書きしたデータ構造を与えて、上書きした設定状態とconfig_box()適用後の実際の設定状態の一致を検証。

唯一の注意点は、config_box() は装置の多くの設定状態を読み書きするので、数秒程度ではあるが、それなりの時間がかかることである。

#### ループバック用の周波数設定に対応
ADCのCDDCの周波数設定をDACのCDUC設定と厳密に一致させる機能を config_port() に実装した。
具体的な方法は、ADCのconfig_box() あるいは、config_port() に渡す設定データ内で、cnco_freq で周波数を指定する代わりに、
cnco_locked_with で周波数を一致させたい出力ポートを指定すればよい。
なお、config_box() を用いた複数ポートの一括設定では、全ての出力ポートが入力ポートに先立って設定されるようになっているので、最新の出力ポートのCNCO周波数
が入力ポートに設定される。

### 波形発生・取得
基本、WaveSubsystemで定義しているメソッドを、port, channel, runit といった Boxレイヤの言葉で記述した引数で呼べるようになっている。

- initialize_all_awgs(): 追加 (wssの同名APIをリダイレクト)
- load_cw_into_channel(): 既存API (start_channel_with_cw) の一部と代替。これとstart_emission() を組み合わせる。
- load_iq_into_channel(): 同上
- prepare_for_emission(): 追加 (wssのclear_before_starting_emission へリダイレクト）
- start_emission(): 追加
- stop_emission(): 追加
- simple_capture_start() : 追加 (wssの同名APIをリダイレクト）

今後、DSP機能などの現在未対応の機能を取り込んでいくにあたり、高度な設定内容を簡便に扱うデータ構造などが導入される予定である。
ここらへんについては、今後のアップデートで大幅な機能増強が行われるが、互換性が大きく崩れることはないと考えている。

## VirtualPort層
まだ、ライブラリとしての実装は無いがクラス定義の[サンプル](../testlibs/general_looptest_common_updated.py)がある。
この例はBoxの使用法の理解にも有用であるが、一括設定の点などにおいて、必ずしも理想的な実装になっていないので注意が必要である。
実際のユースケースに近いのは、おそらく、まず config_box() で各制御装置を設定し終えた後に、波形の生成・取得だけを行う実験フェーズがある、といった構成になるべきであろう。
Box自体は、そのようなユースケースに対応できる設計になっているが、サンプルとしてはそのようになっていないことに注意が必要である。
なお、今後の開発でも、量子計算機の運用・実験の各サイクルで必要な機能を分類し、APIの構成に反映していくつもりである。

Boxを介さないVirtualPortの初期実装も残っているが、これは順次、Boxを使った更新版に置き換わっていく予定である。
