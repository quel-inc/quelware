# ic_config-0.8.x から ic_config-0.10.x への移行について

## 能書き
box層で公開する波形生成・取得のAPIを高レベルなものに置き換え、使い勝手を大きく改善することを目論んだ。
これを実現するために、wave subsystem (wss)のファームウェアを制御する低レイヤのライブラリを従来の [`e7awg_sw`](https://github.com/e-trees/e7awg_sw/tree/simple_multi_standard) 
から新規開発の [`e7awghal`](https://github.com/quel-inc/quelware/tree/main/e7awghal) に挿し替え、quel_ic_config のwss層をほぼ全面的に書き直した。
`e7awghal`パッケージは、`e7awg_sw`パッケージの下半分と、`quel_clock_master`パッケージを統合したパッケージである。
ファームウェアの変更に対し、柔軟に対応できるよう設計されている。
wss層は、`e7awg_sw`パッケージの上半分に加え、今回Boxレイヤで公開している高レベルAPIの主要な部分を実装している。
box層は、波形生成・取得に関しては、このWSS層を使いやすく整形して公開する比較的薄い層にとどまるが、css層とのシームレスなリソース管理も行っている。

今回の変更点はそれなりに大規模だが、波形生成および取得に関わるAPIは、だいたい誰が作っても似通った作りになるので、置き換えは変更点の大きさに比べると容易めになると考える。
box層のwssの操作に関わるAPIは、おおっざぱに、初期化、情報参照、設定、波形生成・取得の4つに分類できる。
初期化と情報参照のAPIは、数が少ないので、まるまる置き換えても多寡がしれているので、先に残りの2つについて説明する。

### 設定系API
設定系のAPIは、0.8.xからの変更点は見かけ上は大したことがない。
0.8.x で試験的に提供している `Quel1BoxWithRawWss`クラスの `config_channel()` が `e7awgsw.WaveSeq` を、
`config_runit()` が `e7awgsw.CapParam` をそれぞれ引数に取るが、これと類似のAPI設計を採用している。
0.10.xは、e7awgswに依存しないので、パラメタのオブジェクトが e7awghal が定義するデータ構造に変更されているが、内容についてシンプルな対応関係が付く。
そういう意味で、コードの変更はさほど大きくならないかもしれない。

しかし、波形データの扱いに関する基本的な発想が変更されているので注意が必要である。
0.8.x では波形生成の度にWaveChunkが握っているiqデータの配列を、制御装置にアップロードしてから波形生成していた。
この設計は、シンプルなことをする限りではうまく使えるが、将来を考えると心許ない。
たとえば、シーケンサが古典if式を実装する場合には、制御装置のメモリ上に複数の波形データを配置しておくべきだ。
また、シーケンサがゲートに対応する波形データを組み合わせて出力波形を合成する機能を実現するためにも、同様の必要性が生じる。

そこで、制御装置内のメモリにある波形データを明示的に管理する仕組みを導入した。
0.10.x では、まず、波形データを装置にアップロードしてから、0.8.x と同様の手順を行うことになる。
- 「出力用の波形データに名前を付けて制御装置のメモリ上に配置する」APIを、波形生成の従来APIから切り離した。
  - チャネルごとに波形ライブラリを持つ。メモリがあるかぎり、任意個の波形データを登録できる。

波形取得についても、同じ発想で波形データ管理に変更を加えている。
これは、設定APIの外観には全く影響を与えないが、メモリ管理の話のついでで、概要を説明しておく。
0.8.x では、波形取得終了後直ちにホストが取得波形のiqデータをダウンロードする作りになっていた。
0.10.x では、波形取得終了後、明示的に波形データをダウンロードするまでは、装置内のメモリにデータが保存される。
この変更は、将来の本格的なシーケンサの実現には不可欠であるが、現状でも、波形取得と量子実験をパイプライン化できるメリットがある。
- キャプチャAPIの返り値が、波形データの読み出し＋パーザの機能を持つオブジェクトになっており、このオブジェクトを叩いて、データをダウンロードする。
  - メモリが溢れない限り、装置側のバッファにデータを置いておける。
  - CapParam設定時に必要量のバッファが自動的に確保され、ダウンロード時完了とともに制御装置内のバッファは自動開放されるので、特にバッファの管理を意識しなくてよい。 
  - （キャプチャAPI時に設定してあるCapParamとパーザが連動しており、構造化済みのデータを取得できる。）

### 波形生成・取得API
波形発生及び取得系のAPIは、3つのAPIに集約した。
- 波形生成開始（即時 or 時刻指定）
- 即時の波形取得開始
- 波形生成と波形取得の同時開始（即時 or 時刻指定）

この3つ全てのAPIは非同期設計になっており、返り値にFutureオブジェクトを返す。
波形取得のAPIは0.8.xでも非同期になっていたが、波形生成も非同期化して、読み出し動作（読み出し信号を出力して、反射をキャプチャする）という
典型的動作を簡単かつ安全に実現できるようにした。
特に、自動的に適切なタイムアウトを設定し、ファームウェアの異常を迅速に検出し、エラー発生時の回復処理を自動的に行う。

まず、大幅な変更がある波形生成のAPIについて説明する。
0.8.x では、start と stop で波形を出したり止めたりする作りになっていたが、これは、あまり良い設計では無かった。
波形を強制的に止めるのはノミナルな動作ではないので、実は stop はあまり大事ではない。
また、start 時に指定したチャネルと、stop 時に指定するチャネルの対応管理も煩雑だった。
そこで、波形発生開始APIがFutureオブジェクトを返す形に非同期化した。
Futureオブジェクトに対する次の2つのメソッドを用いて、ハードウェアの状態を把握および制御できる。
- `result()` で波形発生が終了するまで待つ。
- `cancel()` で波形発生を強制終了させる。

なお、通常の`concurrent`モジュールの`Future`はスレッドの実行前に限り `cancel()` 可能という意味論だが、ここではそれを拡張して、波形生成を途中で停止できるような拡張をしている。
`cancel()` の処理が正しく行われたことの確認は、`result()`で行う。
その場合、`CancelledError` の例外が発生するので、適切に例外のハンドリングをする必要がある。

波形取得APIについては、上記第3項目の「波形生成と波形取得の同時開始」を1つのAPIで安全かつ簡単に行えるようになったのが、一番の目玉であると思う。
実用上は、上記第2項目の即時の波形取得は使い所が少ないと思う。
また、あまり使うことはないだろうが、波形取得も `cancel()`可能になった。
同時開始の設計意図は、キャプチャモジュールのトリガ設定の管理を自動化し、開始前の設定ミスの検出及び防止や、異常終了時やキャンセル時の整合的な復帰処理の実現にある。
この意味で、このBox層のAPIは、WSS層の波形生成と波形取得のAPIを単純な組み合わせではない。
基本的に WSS層のAPIの直接使用は非推奨であるが、仮にそうする場合には、両APIの挙動と相互作用の詳細を理解する必要があるので、いろいろと面倒である。

## WSSを操作するBox層のAPIのリスト
### 初期化
#### `Quel1Box.initialize(self) -> None`
Box全体の初期化時に、WSSの初期化が含まれている。WSSは初期化しないで使うと不整合が発生するので、Boxオブジェクトを作成後すぐに呼ぶこと。

#### `Quel1Box.initialize_all_awgunits(self) -> None`
波形生成全体の初期化をする。波形生成が全て停止する。

#### `Quel1Box.initialize_all_capunits(self) -> None`
全てのキャプチャユニットを初期化する。実行中のキャプチャを停止、設定中のパラメタ及びトリガの設定を解除する。

### 情報取得
#### `Quel1Box.get_current_timecounter(self) -> int`
制御装置が管理している時刻カウンタの値を取得する。
装置のリンクアップ開始時から、125MHzのtick をカウントしている。

#### `Quel1Box.get_latest_sysref_timecounter(self) -> int`
全装置に共通で入力している 62.5kHz のSYSREF信号の最新のエッジを検出したときの時刻カウンタの値を返す。
全装置の時刻カウンタが数ミリ秒の精度で合っていれば、この値の2000(= 125MHz / 62.5kHz)の剰余を比較することで装置間の時刻のずれを8nsの精度で検出できるはずだった。
現状では、回路実装上の制約でSYSREFのエッジが鈍っており、単発では +/-3クロック程度の誤差が発生するので、平均化して使うことで、それに近い精度での時刻同期をできる。

### 設定
#### `Quel1Box.config_channel(self, port: Quel1PortType, channel: int, *, fnco_freq: Union[float, None] = None, awg_param: Union[AwgParam, None] = None) -> None`
指定チャネルに `awg_param: AwgParam` を与えて、生成する波形シーケンスを設定する。
`AwgParam` については、別途説明

#### `Quel1Box.config_runit(self, port: Quel1PortType, runit: int, *, fnco_freq: Union[float, None] = None, capture_param: Union[CapParam, None] = None) -> None`
指定のrunit に、`capture_param: CapParam` を与えて、受信サンプル数や繰り返し回数などの各種設定を行う。
`CapParam` については、別途説明。

### 波形データ登録
#### `Quel1Box.get_names_of_wavedata(self, port: Quel1PortType, channel: int) -> set[str]`
指定チャネルに登録済みの波形データの名前一覧を取得する。

#### `Quel1Box.register_wavedata(self, port: Quel1PortType, channel: int, name: str, iq: npt.NDArray[np.complex64], allow_update: bool = True, **kwdargs)`
指定チャネルに、`name`で指定した名前の波形データ`iq`を登録する。
`allow_update`は同名のデータがあった場合に上書きする。
デフォルトで True になっているが、厳格な管理をしたい場合には False にすべきである。
キーワード引数はアロケータに渡るパラメタだが、有用なものは今のところ何もない。

#### `Quel1Box.has_wavedata(self, port: Quel1PortType, channel: int, name: str) -> bool`
指定チャネルに、`name`で指定の名前の波形データがあれば `True`、なければ `False` を返す。

#### `Quel1Box.delete_wavedata(self, port: Quel1PortType, channel: int, name: str) -> None`
指定チャネルの `name`で指定の名前の波形データを削除する。存在しない場合には `ValueError`を発生する。

### 波形発生・取得
#### `Quel1Box.start_wavegen(self, channels: Collection[Tuple[Quel1PortType, int]], timecounter: Optional[int] = None) -> AbstractStartAwgunitsTask`
指定の複数のチャネルで波形発生を開始する。`timecounter`が Noneなら即時開始、そうでなければ指定の時刻に開始する。
返り値は`AbstractStartAwgunitsTask`クラスのオブジェクトで、`concurrent.Future`と同じインターフェースを持つ。
`result()` の返り値は`None`なので、波形発生の終了を確認するのに用いる。

`cancel()`で、生成中の波形を止めることができるが、注意点がある。
時刻指定で開始した場合のキャンセルが、波形発生が始まる前に行われた場合、シーケンサキューへ予約の破棄の依頼をする。
現状のシーケンサの実装上の制約で、キュー内の全ての予約の一括破棄しかできない。
したがって、複数の予約が存在する場合、全てが破棄されてしまう。
この制約から、`start_wavegen()`をはじめとする、時刻指定の波形生成の予約は同時には1つに限定した運用をするべきである。

複数の予約が同時に存在することを禁止はしていないが、複数予約をできるようにすることよりも、ひとつの予約を安全に遂行することを優先した設計になっている。
今後も、ファームウェア側で適切な支援の仕組みが整備されるまでは、予約は1つのポリシーを設計に適用していく。

#### `Quel1Box.start_capture_now(self, runits: Collection[Tuple[Quel1PortType, int]]) -> BoxStartCapunitsNowTask` 
指定の複数のrunitで、即時の波形取得を開始する。
返り値の`BoxStartCapunitNowTask`クラスのオブジェクトは、例によって `concurrent.Future`と同じインターフェースを持つ。
`result()` の返り値は、`CapIqDataReader`クラスのオブジェクトである。
このオブジェクトが、装置からデータをダウンロードし、構造化したデータをユーザに提供する。
これについては次節で例を使って説明する。

#### `Quel1Box.start_capture_by_awg_trigger(self, runits: Collection[Tuple[Quel1PortType, int]], channels: Collection[Tuple[Quel1PortType, int]], timecounter: Optional[int] = None) -> tuple[BoxStartCapunitsByTriggerTask, AbstractStartAwgunitsTask]`
指定の複数のrunit と指定の複数のチャネルとを同時に起動し、波形生成と取得を一斉に開始する。
timecounterがNoneの場合は即時開始、そうでなければ指定の時刻カウンタでの開始となる。
即時実行の場合は、 `start_capture_now()` と `start_wavegen()` とを同時に実行したのと等価である。
一方、時刻指定の場合には、両方を一括で設定することが異常検出の観点から好ましいので、このようなAPI設計になった。

1つ目の返り値の`BoxStartCapunitsByTriggerTask` クラスのオブジェクトは、上述の`BoxStartCapunitsNow`クラスのオブジェクトと同じふるまいをする。
2つ目も 上述の`start_wavegen()`の返り値と同じものである。
返り値のオブジェクトは、両方とも `result()`を適用し、少なくとも終了確認をする必要があるが、順番に制約はない。
通常の用途では、波形生成とキャプチャは同時に終わるはずなので、2つのオブジェクトを1つに統合するか迷ったが、APIの構成上、同時に終わることが保証できないので、そのままにしてある。
Boxのさらに上位の層で、波形生成とキャプチャの設定も含めて自動で行う場合には、1つに統合するだろう。

## 関連するデータ構造など
### AwgParamの作成
AWGのパラメタオブジェクトの設定項目の詳細については[e7awgsw](https://github.com/e-trees/e7awg_sw/tree/simple_multi_standard/manuals#3-awg-%E3%82%BD%E3%83%95%E3%83%88%E3%82%A6%E3%82%A7%E3%82%A2%E3%82%A4%E3%83%B3%E3%82%BF%E3%83%95%E3%82%A7%E3%83%BC%E3%82%B9%E4%BB%95%E6%A7%98)
を参照。
パラメタオブジェクトは pydantic.BaseModel に基づいて作られており、使い方はシンプルだが、それぞれのパラメタ内容を理解するが厄介である。
上述の文書で概要を掴んだら、実際にAPIを叩いてみるのが理解の早道だと思う。

波形パラメタの作成と設定の手順を例で説明する。
前提として、波形データを装置内メモリにアップロードしておく必要がある。
ここでは、次のような手順で、ポート(port_idx)のチャネル(channel_idx)に、"cw32676"という名前で、
長さ64サンプルの直流のベースバンド波形が登録済みであると前提する。
```python
cw_iq = np.zeros(64, dtype=np.complex64)
cw_iq[:] = 32767.0 + 0.0j
box.register_wavedata(port_idx, channel_idx, "cw32767", cw_iq)
```
波形データの登録が、チャネル単位になることに注意が必要だ。
また、波形データ長は64サンプルの倍数でなくてはいけない。

この波形データから、WaveChunkを作成し、それをAwgParam内に並べる。
64サンプルの長さの方形波を出力した後に192サンプル休むのを3回繰り返す場合には、次のようなパラメタオブジェクトを作成する。
```python
ap = AwgParam(num_repeat=3)
ap.chunks.append(WaveChunk(name_of_wavedata="cw32767", num_blank_word=192//4, num_repeat=1))
```
無信号時間を指定する単位が、word になっていることに注意。
1 word は 4 IQsample である。
この場合には、`AwgParam`を`num_repeat=1` にして、`WaveChunk`を`num_repeat=3` にしても同じ出力が得られる。
1つのAwgParamに複数のWaveChunkを入れる場合には、全体を繰り返すのが前者、特定のチャンクだけを繰り返すのが後者であることを意識する必要がある。
なお、1つのAwgParamにWaveChunkを16個まで入れることができる。

このオブジェクトを所定のチャネルに設定すればいい。
```python 
box.config_channel(port_idx, channel_idx, awg_param=ap)
```
これで波形発生の準備は完了である。
なお、config_channelで、fnco_freq の設定を同時に行ってもよい。

AwgParamはチャネルとは独立なので、他の "cw32767"という名前の波形データを持っている他のチャネルにも適用可能である。
これをうまく使うと、チャネル間で波形パラメタ設定の処理を共通化できる。
波形データの命名規則を次第で、便利な波形ライブラリを作れると思う。
なお、config_channel() をしたタイミングで、波形データの名前の解決を含む、いくつかのバリデーションを行うことにも注意が必要だ。

### CapParamの作成
キャプチャユニットのパラメタや得られるデータの詳細については[e7awgsw](https://github.com/e-trees/e7awg_sw/tree/main/manuals#4-%E3%82%AD%E3%83%A3%E3%83%97%E3%83%81%E3%83%A3%E3%83%A2%E3%82%B8%E3%83%A5%E3%83%BC%E3%83%AB%E3%82%BD%E3%83%95%E3%83%88%E3%82%A6%E3%82%A7%E3%82%A2%E3%82%A4%E3%83%B3%E3%82%BF%E3%83%95%E3%82%A7%E3%83%BC%E3%82%B9%E4%BB%95%E6%A7%98)
を参照。すごく複雑なので文書で概要を掴んだ後に、実際にAPIを叩いてみて、得られるデータの変化を見るというやり方が、理解が早いかもしれない。

こちらも実際にパラメタを作成する例を示して説明する。
256サンプルの周期を3回繰り返し、各周期で192サンプルのデータ取得、64サンプルを読み飛ばす、場合には次のようなパラメタオブジェクトを作成する。
```python
cp = CapParam(num_repeat=3)
cp.sections.append(CapSection(name="s0", num_capture_word=192//4, num_blank_word=64//4))
```
指定する単位が、word になっていることに注意。
1 word は 4 IQsample である。
e7awgswの説明にあるように、CapSectionは4096個並べることができる。
今回は、CapSectionをひとつだけ入れて、名前を "s0" とした。
この名前は後で、取得した波形データにアクセスするときに使う。

これを所定のrunitに設定すれば、キャプチャを開始する準備は完了である。
```python
box.config_runit(port_idx, runit_idx, capture_param=cp)
```

### CapIqDataReader の API
波形取得のAPIの返り値は、`Future[CapIqDataReader]`と同じインターフェースを持つ。
result() でCapIqDataReaderを取得した後に、次のようにして、制御装置から波形データをダウンロードする。

先ほどの例で作成した`CapParam` で波形取得をした場合、次のような手順で各データにアクセスできる。
```python
task = box.start_capture_now({(port_idx, runit_idx)})  # 波形取得を行う。
rdr = task.result()  # 波形取得完了を待つ。CaqIqDataReader が各runitごとに作成される。
data = rdr[port_idx, runit_idx].as_wave_dict()  # 各runitの波形データをダウンロードする。
# data["s0"] で、"s0"のセクションデータが取れる。
# セクション内は、要素がnp.complex64型の 2次元の配列になっている。
# 最初の軸が繰り返し回数、2番目の軸がサンプル番号なので、shapeが（3, 192) となる。
data0 = data["s0"][0]  # 1回目の繰り返しの192サンプルのデータ 
data1 = data["s0"][1]  # 2回目の繰り返しの192サンプルのデータ 
data2 = data["s0"][2]  # 3回目の繰り返しの192サンプルのデータ 
```

CapIqDataReaderの主要APIは以下のとおり。
この他に、構造化前の生データを見たり、データのダウンロードだけをあらかじめ行うAPIもある。
これらの詳細はソースコードを参照して頂きたい。

#### `CapIqDataReader.as_wave_dict(self) -> dict[str, npt.NDArray[np.complex64]]`
取得したデータのセクション名で対応するデータを引く辞書として返す。
セクション内は、2次元のNDArrayになっており、最初の軸が繰り返し回数、2番目の軸がサンプル番号となる。
（セクション名は、CapParamにCapSectionを追加するときに指定したもので参照する。）

#### `CapIqDataReader.as_wave_list(self) -> list[npt.NDArray[np.complex64]]`
as_wave_dict() とほぼ同じだが、セクションの参照を名前ではなく順序で行う。
CapSectionは名前を付けないこともできるが、その場合には、こちらのAPIだけしか使えない。

#### `CapIqDataReader.as_class_dict(self) -> dict[str, npt.NDArray[np.uint8]]`
CapParam で classifier を有効にした場合で、セクションの参照を名前で行う場合に使う。

#### `CapIqDataReader.as_class_list(self) -> list[npt.NDArray[np.uint8]]`
CapParam で classifier を有効にした場合で、セクションの参照を順序で行う場合に使う。

## サンプルスクリプト
[このサンプル](../scripts/simple_timed_loopback_example_quel1se_riken8.py)が、設定から波形生成及び取得までの一連の流れを
理解するのに適している。
QuEL-1 SE RIKEN8モデルの制御装置で実行可能である。
動作は次のとおり。
- 指定の制御装置の各ポートの周波数や強度などの設定を一括して行う。
   - 全ポートの設定データをdictとして保持している。jsonを引数に取れる同様のAPIもある。
- 波形パラメタやキャプチャパラメタを作成して、各ポートを設定する。
- 0.1秒後に全出力ポートと全入力ポートを一斉に起動。
   - 正確には、全出力ポートのchannel-#0 と、全入力ポートの runit-#0 とを一斉起動。
- 動作終了後にキャプチャデータを取得して、グラフに表示。

[GETTING_STARTED.md](./GETTING_STARTED.md#コード例)により具体的な説明がある。
