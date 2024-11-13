# サンプルコードの使い方
[`scripts`](../scripts) ディレクトリ内のサンプルコードの内容と使い方を説明する。

## 準備
[GETTING_STARTED.md](GETTING_STARTED.md) の手順にしたがって、実行環境を構築する必要がある。
サンプルコードの依存パッケージを解消するために `requirements_dev_addon.md` もインストールしておくとよい。

## quel1_monitor_synchronized_output.py
上述の`quel1_check_all_internal_loopbacks.py`は、QuBE OU Type-B のような入力ポートが一切ない装置では使うことができない。
これに対応するために、測定対象の出力ポートの出力波形を、別の機体の入力ポートで測定するスクリプトを用意した。

装置を跨いで波形生成とキャプチャを同期的に行うには、キャプチャの開始を同じ機体の任意の出力ポートの出力開始でトリガする必要がある。
したがって、スクリプトは3つのポート、つまり、入力ポート、動作確認対象の出力ポートに加えて、トリガポートを必要とする。
この際に、トリガポートと出力ポートをコンバイナを介して入力ポートへ接続するのが理想的である。
スクリプトは、トリガポートからトリガと同時に64サンプルのパルスを発生し、出力ポートから256サンプル遅れで128サンプルのパルスを発生する。
この両方のパルスをキャプチャすることで、装置間のタイミングを詳細に確認することもできる。

### 制限事項
QuEL-1 SE はこのスクリプトではサポート外である。
いずれ、QuEL-1 SE用の同等のスクリプトも提供する予定である。

### 使用例
QuEL-1 Type-A である10.1.0.58のポート5で、QuEL-1 Type-B である10.1.0.60 のポート4の出力波形をキャプチャする場合を考える。
ポート5のキャプチャ開始には、同機のポート2を使用する。
次のコマンドの実行に先立って、10.1.0.58のポート2と10.1.0.60のポート4を2対1のコンバイナで合波し、10.1.0.58のポート5に接続しておく。
```shell
PYTHONPATH=. python scripts/quel1_monitor_synchronized_output.py --ipaddr_clk 10.3.0.13 \
--ipaddr_wss_a 10.1.0.58 --boxtype_a quel1-a --input_port 5 --trigger_port 2 \
--ipaddr_wss_b 10.1.0.60 --boxtype_b quel1-b --output_port 4
```

次のようなコンソールログとキャプチャ波形とが得られるはずである。
```
2024-05-09 23:54:38,664 [INFO] testlibs.general_looptest_common_updated: number_of_chunks: 2
2024-05-09 23:54:38,664 [INFO] testlibs.general_looptest_common_updated:   chunk 0: 64 samples, (497 -- 561)
2024-05-09 23:54:38,664 [INFO] testlibs.general_looptest_common_updated:   chunk 1: 130 samples, (746 -- 876)
```

![monitor_sync](images/monitor_sync.png)

1つ目のパルスがトリガポートからのパルスで、キャプチャ開始から497サンプル目に取得できている。
2つ目の幅が広いパルスが動作確認対象の出力ポートからのパルスで、249 (= 746 - 497) サンプル遅れでキャプチャできている。
両方のポートから入力ポートまでの経路長が同じで、かつ、2台の制御装置が完全に同期しており、かつ、波形発生のタイミングが装置間でアラインメントが取れていれば、256サンプル遅れになるはずである。
経路長はほぼ同じにしており、かつ、時刻カウンタの補正を施した状態でなので、タイミングのずれの原因は、波形発生のタイミングがアラインメントの問題であると推測できる。
波形発生のタイミングは64サンプルに単位でしかずらせないので、10.1.0.60に与える波形データの最初に6ないし7サンプルのゼロを挿入すればタイミングが揃うはずだ。

## quel1_monitor_synchronized_output_advanced.py
先述の `quel1_monitor_synchronized_output.py` を仕立て直して、装置間のスキュー調整を手動で試すためのスクリプトとしたものである。
元になったスクリプトが使用している `BoxPool.emit_at()` の `time_count` 引数は波形発生のタイミングを相対的に指定するが、
装置間同期に用いているSYSREFクロックのエッジのタイミングを考慮して、波形発生の絶対タイミングを決定している。
具体的には、time_count が 0 のときに、SYSREFクロックのエッジに対して、波形チャンクの単位サイズ(64サンプル、128ns, 125MHzのシステムクロックで16クロック)の倍数時間後のタイミングに波形要求のトリガを出す。

このスクリプトでは、波形発生のタイミングをこの境界に対し、-8クロックから8クロックまでのオフセットを与えながら元のスクリプトの手順を繰り返し実行し、グラフとレポートを表示する。
なお、オフセットが0の場合が、元のスクリプトと同じ動作となる。
使い方も元のスクリプトと同じであるが、いくつか引数が追加されているので、次のコマンドを実行した場合グラフを見ながら、順を追って説明していく。

```shell
PYTHONPATH=. python scripts/quel1_monitor_synchronized_output_advanced.py --ipaddr_clk 10.3.0.13 \
--ipaddr_wss_a 10.1.0.58 --boxtype_a quel1-a --input_port 5 --trigger_port 2 \
--ipaddr_wss_b 10.1.0.60 --boxtype_b quel1-b --output_port 4
```

![skew](images/skew.png)

ここで、オフセット値（グラフ中では`time_to_start:`の後ろの値）のある値を境に64サンプルずれる現象が発生することが分かる。
この「ずれ」はトリガのオフセット値の16の剰余に依存して決まる。
しかし、「ずれ」が起こる具体的なオフセット値は、リンクアップをする度に変化するので、キャリブレーション時に考慮する必要がある。
特に「ずれ」の切り替わり点では、期待どおりのタイミングで波形発生がはじまるか、あるいは、64サンプル遅れになるかが、低い確率ではあるがバタつくことがあるので、
境界すれすれのタイミングでトリガを掛けるのは避けるべきである。
上の図の場合では、オフセットを -3 あたりでトリガを掛けるのが安全であることが分かる。

グラフの横軸の原点は、測定対象ポートからのパルス発生の期待値に合わせてある。
つまり、トリガポートのパルスのエッジから`--output_delay`サンプル（デフォルト値は256) だけ遅れたサンプルを0としている。
オフセット値が-7から0 の場合に注目すると、期待値よりもやや早く波形発生が始まってしまっている。
一方で、オフセット値が1から8の場合では大きく遅れているので、-7から0の方が期待どおりの挙動に近い。
コマンドのコンソールログに、これらについて詳しい情報を表示している。
```shell
2024-05-10 20:35:01,611 [INFO] root: expected start time of the pulse: 753.0
2024-05-10 20:35:01,612 [INFO] root: actual start time of the pulse at the offset of -8: 811.0
2024-05-10 20:35:01,612 [INFO] root: actual start time of the pulse at the offset of -7: 747.0
2024-05-10 20:35:01,612 [INFO] root: actual start time of the pulse at the offset of -6: 747.0
2024-05-10 20:35:01,612 [INFO] root: actual start time of the pulse at the offset of -5: 747.0
2024-05-10 20:35:01,612 [INFO] root: actual start time of the pulse at the offset of -4: 747.0
2024-05-10 20:35:01,612 [INFO] root: actual start time of the pulse at the offset of -3: 747.0
2024-05-10 20:35:01,612 [INFO] root: actual start time of the pulse at the offset of -2: 747.0
2024-05-10 20:35:01,612 [INFO] root: actual start time of the pulse at the offset of -1: 747.0
2024-05-10 20:35:01,612 [INFO] root: actual start time of the pulse at the offset of 0: 747.0
2024-05-10 20:35:01,612 [INFO] root: actual start time of the pulse at the offset of 1: 811.0
2024-05-10 20:35:01,612 [INFO] root: actual start time of the pulse at the offset of 2: 811.0
2024-05-10 20:35:01,612 [INFO] root: actual start time of the pulse at the offset of 3: 811.0
2024-05-10 20:35:01,612 [INFO] root: actual start time of the pulse at the offset of 4: 811.0
2024-05-10 20:35:01,612 [INFO] root: actual start time of the pulse at the offset of 5: 811.0
2024-05-10 20:35:01,612 [INFO] root: actual start time of the pulse at the offset of 6: 811.0
2024-05-10 20:35:01,612 [INFO] root: actual start time of the pulse at the offset of 7: 811.0
2024-05-10 20:35:01,612 [INFO] root: actual start time of the pulse at the offset of 8: 811.0
```

このログから、パルスの発生タイミングの期待値(753サンプル目)よりも、6クロック早くパルスが出ていることが分かる。

そこで、`--output_delay_delta 8` を引数に追加すると、波形データの先頭にゼロを8サンプル追加して、誤差を減らすことがきる。
サンプル数が4の倍数でなければいけないので、`4` か `8` か好ましい方を選ぶことになる。
その結果、だいたい期待どおりのタイミングで波形発生ができるようになる。

というように、10.1.0.58 と 10.1.0.60 との波形発生のタイミングをSMA端で合わせる場合には、10.1.0.60の波形データにゼロを6サンプル挿入すればよいことが分かる。
また、トリガタイミングを -3 ずらしておくと、安心である。
このことは、次のコマンドで確認できる。
```shell
PYTHONPATH=. python scripts/quel1_monitor_synchronized_output_advanced.py --ipaddr_clk 10.3.0.13 \
--ipaddr_wss_a 10.1.0.58 --boxtype_a quel1-a --input_port 5 --trigger_port 2 \
--ipaddr_wss_b 10.1.0.60 --boxtype_b quel1-b --output_port 4 \
--delay_output_delta 8 --trigger_offset -3
```

`time_to_start: 0`のグラフが、期待通りのタイミングで波形発生できるオフセットの中央にきていることが見て取れる。

![skew_corrected](images/skew_corrected.png)

## getting_started_example.py
[GETTING_STARTED.md](GETTING_STARTED.md) にて説明しているので、そちらを参照していただきたい。
