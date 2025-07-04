# 更新リスト

## v0.10.3 (Public Release)
- uvによるパッケージ管理を導入。
- quel1_parallel_linkupにignore_crc_error_of_mxfe等のオプションを追加。
- Matplotlibのバックエンドの明示的な指定を排除。
- 負周波数におけるNCOのFTW設定の不具合を修正。
- quel1_dump_port_configオプションが正しく動作するように修正。
- quel1_staging_toolで使用するファームウェアを別途配布する形式に変更。
- 不要コード削除等のリファクタリング。

## v0.10.2 (Public Release)
- quel1_parallel_link コマンドのバグ修正。
  - 対象装置で特定の異常が発生した際に、コマンドが異常終了する問題を解消。
  - パフォーマンスを調整。
- 制御装置との通信のタイムアウト時間を調整。
- quel1_check_all_internal_loopbacks コマンドの quel1se-fujitsu11-b用の設定を修正。

## v0.10.1 (Public Release)
- WaveSubsystem(wss)を挿し替え 
  - e7awgsw との依存を解消。
  - quel_clock_master も代替。
  - 500MHz情報帯域のチャネルを生のままユーザに提供する形式のAPIは固定。
  - 詳細は[こちら](./docs/MIGRATION_TO_0_10_X.md)を参照。
- 排他的使用を支援するロック機構を実装。
  - 従来のファイルロックと類似の仕組みをBox単位で実装。
    - ロックファイル用のディレクトリ(/run/quelware)をNFSで共有することで、複数ホスト間でのロックを実現。
    - ロックの仕組みを持たない古いバージョンのソフトウェアからのアクセスを防ぐことができないので注意。
      - したがって、quelware-0.8.x系との混在は排他制御の確保に障害が発生するので基本避けるべき。
  - 詳細は[こちら](./docs/BOX_EXCLUSIVE_ACCESS_CONTROL.md)を参照。
- quel1_parallel_linkup コマンドを更新。
  - `--background_noise_threshold` オプションの引数を再リンクアップ時だけでなく、再接続時にも適用。
  - 再接続時の許容最大ノイズ値を 1536 から 4096 に増加。外部接続機器からのノイズ起因で、再リンクアップの実施判定が出にくいようにした。
  - `--hard_reset_wss` オプションでAWGユニットとCaptureユニットのハードウェアエラーフラグをクリアできるようにした。
  - clockmaster の情報を書けるように拡張したバージョン2のファイルフォーマットに対応（後述）。
- quel1_sync と quel1_syncstatus コマンドを導入。これらで、装置間同期の実施と状態確認を行う。
  - quel1_parallel_linkup と同じファイルを引数に取るので、リンクアップと時計合わせを一元的に行える。
- quel1_linkstatus コマンドを修正。背景ノイズのチェックが正しく最終判定に適用されるようにした。また、background_noise_threshold オプションを追加。
- quel1se-fujitsu11-a 及び -b の正式サポートを開始。
- ADI製のAD9082の低レイヤライブラリを v1.7.0 にアップデート
  - それに合わせてリンクアップ手順を最適化。
- SIMPLEMULTI_CLASSIC版ファームウェアの対応停止。
- FEEDBACK_EARLY版ファームウェアの対応一時停止。
- ping3への依存を解消。

### v0.9.6
- v0.9.5 の不具合を修正。

### v0.9.5
- QuEL-1 SEのロック付きCSSファームウェアに対応
- [Quel1AnyConfigSubsystem](./src/quel_ic_config/quel1_any_config_subsystem.py) をリファクタ
  - Quel1AnyBoxConfigSubsystemを廃止。Quel1AnyConfigSubsystemに互換的に移行可能。
- E7ResoruceMapper を全面改修。
  - CSSの境界を FDUC, FDDC として整理。
  - WSSの下位レイヤの挿し替えの準備対応。
- SIMPLEMULTI_CLASSIC への対応を廃止。 

### v0.9.4
- `Quel1Box`クラスのconfig系APIのポート指定の方法を拡張。従来の `port` と `subport` を別個の引数として与える形式に加えて、`port`引数に (port, subport)のタプルを与える形式も使用可能とした。
- `Quel1Box`クラスに、実行時の設定パラメタをjsonファイルに保存したり、jsonファイルから読み出したりするAPIを実装。
  - `dump_box_to_jsonfile()`
  - `config_box_from_jsonfile()`
- `Quel1Box`クラスに、イントロスペクションAPIを追加。
  - 各種ポートのリストの取得。
  - 各出力ポートのchannelのリストを取得。
  - 各入力ポートのrunitのリストを取得。
  - 各入力ポートのループバックで取得可能な出力ポートのリストを取得。
- config系のファームウェア v1.2.1 に対応
- 期待の範囲内の異常で表示されていたエラーログを警告ログに降格。
- `quel1_clear_sequence`コマンドを追加。未実行の時刻指定型コマンドをキャンセルするときに使う。
- QuEL-1 SE の 11GHz版に暫定対応。

### v0.9.3
- ソースコードのファイルツリーの大規模変更。
- AD9082の有理数FTW (dual modulus モード) に対応。
  - 任意の整数Hzの周波数を誤差なく設定可能。（たとえば、1GHzを設定すると生じていた42uHz程度の誤差は、もはや発生しない。）
  - `Quel1Box.allow_dual_modulus_nco` を `False` に設定すると従来どおりの動作になる。

### v0.9.2
- タイムトリガによる波形発生の動作確認スクリプトを整備。
- モニタ系を備えた各制御装置モデルについて、内部ループを使った波形生成の動作確認用のスクリプトを整備。
- `testlibs.general_looptest_common_updated.py` の設定内容のデータ構造と一部のAPI設計を改善。
- テストコード及びサンプルコードのQt5依存を除去し、代わりにGtk3を使用。

### v0.9.1
- QuEL-1 SE Riken-8 の設定系のファームウェアの更新とその対応。
- リンクアップパラメタの改善とバックグラウンドキャリブレーションの適用。
  - QuEL-1のリンクアップにおいても、デフォルトでJESD204C準拠のリンクキャリブレーションが適用となる（`--use_204c` はもはや必要ない。）
  - 従来動作を強制するためのオプションとして`--use_204b`オプションを新設。
  - バックグラウンドキャリブレーションを非適用とするためのオプションとして、`--nouse_bgcal` を新設。
- 並列自動リンクアップツールを追加。
  - yamlファイルに記述した機体について、リンク状態が異常であった場合に再リンクアップする。

### v0.9.0
- e7awgsw の `WaveSequence` と `CaptureParam` を一時的に借り受けて、`Quel1Box` から各種設定を可能とした。
  - `Quel1Box.config_channel()` に `wave_param` 引数を追加。
  - `Quel1Box.config_runit()` に `capture_param` 引数を追加。
  - 従来の次のAPIは、これらの仕組みとは併用できない。なぜならば、これらのAPIは`wave_param`や`capture_param`を上書きしてしまうので。
    - `Quel1Box.load_cw_into_channel()`, `Quel1Box.load_iq_into_channel()`
    - `Quel1Box.simple_capture_start()`, `Quel1Box.simple_capture()`
  - `capture_param` を与える設定法と整合的な波形取得用のAPIとして、`Quel1Box.capture_start()` を追加。 

## v0.8.8 (Public Release)
- QuEL-1 SE Riken-8 を公式サポート開始
- `Quel1Box`のいくつかのAPIのエラーメッセージを改善
- `Quel1Box.config_box()` の `fullscale_current` の同一性確認の不具合を修正
- 恒温システムの状態確認とリセットのコマンドを追加
- 恒温システムのウォームアップタイマ設定の不具合を修正
- `quel1_linkstatus`コマンドの `--ignore_crc_error_of_mxfe` オブション付加時のレポート文言を変更
- ドキュメントを更新

## v0.8.7 (Public Release)
- ドキュメントを更新。
- 一部のINFOレベルのログをDEBUGレベルに下げた。
- `load_cw_into_channel()` と `load_iq_into_channel()` のブランク長の指定をワード単位からサンプル単位に変更。
- `simple_capture_start()` の `delay` の指定をワード単位からサンプル単位に変更。引数名を`delay`から`delay_samples`に変更。
- `load_*_into_channel()` の `num_wait_words` を `num_wait_samples` に名称変更し、ワード単位からサンプル単位での指定に変更。

### v0.8.6
- v0.8.5で混入したバグ修正

### v0.8.5
- 入力ポートのCNCO設定を出力ポートのCNCO設定で制約し、正確に同じ周波数設定をする機能を追加。
  - `Quel1Box.config_port()` に `cnco_locked_with` 引数を追加。制約元の出力ポートを与えて使う。
  - `Quel1Box.config_box()` で複数のポートを同時に設定する場合には、`cnco_locked_with` の制約元ポートが、制約を受けるポートよりも必ず先に設定される。

### v0.8.4
- Boxレイヤのconfig / dump 系のAPIを整理。
  - Boxの階層構造に基づいたAPI 
    - `config_box` / `config_port` / (`config_channel`, `config_runit`)
    - `dump_box` / `dump_port`
  - RFスイッチの操作だけを行うAPI
    - `config_rfswitches` / `config_rfswitch`
    - `block_all_output_port`, `pass_all_output_port`
    - `dump_rfswitches` / `dump_rfswitch`
  - 削除されたAPIとその代替APIは次のとおり。
    - `open_rfswitch`, `close_rfswitch` --> RFスイッチ系のAPI
    - `dump_config` --> `dump_box` （改名）

### v0.8.3
- Boxレイヤに残っていた `rchannel` の概念を消し去り、`runit` に置き換え。
   - `config_channel()` は入力チャネルには使えなくなり、代わりに、`config_runit()` を使うこと。
   　　- 概念の明確化と将来の変更に向けての布石。
   　　- 複数の runit がひとつの rchannelを共有している。Boxレイヤでは rchannel は陽に見せない方針。
   - `dump_config()` の返り値の形式が変更。同様に、`config_box()` の引数の形式も変更。
      - `channel` が `runit` に置き換え。
- バグ修正
   - `Quel1Box.easy_stop_all()` のバグ修正。
   - 一部の機種で `Quel1Box.dump_config()` が全てのポートを列挙しない問題を修正

### v0.8.2
- Quel1Boxの波形生成・取得のとりあえずのAPIを整理。（今後、DSP関連などの高度な機能を使うためのAPI拡張をする）
   - `initialize_all_awgs()`: 追加 (wssの同名APIをリダイレクト)
   - `load_cw_into_channel()`: 既存API (`start_channel_with_cw()`) の一部と代替。これと`start_emission()` を組み合わせる。
   - `load_iq_into_channel()`: 同上
   - `prepare_for_emission()`: 追加 (wssの`clear_before_starting_emission()` へリダイレクト）
   - `start_emission()`: 追加
   - `stop_emission()`: 追加
   - `simple_capture_start()` : 追加 (wssの同名APIをリダイレクト）
- Config系ファームウェアの更新に対応。
   - v1.0.2 
- Boxレイヤと正しくつながるVirtualPortレイヤのプロトタイプを追加

### v0.8.1
- v0.8.0 でプッシュし忘れたファイルを追加（もうしわけありません）。

### v0.8.0
- `SimpleBox` を `Quel1Box` に改名し、simple_box.pyで実装していたヘルパ関数群を整理。
  - `init_box_with_*()` の返り値からBoxオブジェクトを削除。開発専用のAPIとなり、通常用途の使用は非推奨。
     - 代替手段として、`Quel1Box`に新しいクラスメソッド `create()` を新設。
  - `reconnect()`, `linkup()` を `reconnect_dev()`, `linkup_dev()` と改名。開発専用のAPIとなり、通常用途での使用は非推奨。
     - 代替手段として、`Quel1Box`の新しいメソッド `reconnect()` と `relinkup()` を新設。
     - `reconnect_dev()`, `linkup_dev()` は`Quel1Box`を引数に取れません。
  - ご迷惑をおかけします。詳しくはマイグレーションガイドを参照してください。
- `Quel1Box`に、装置のIC群を一括で設定するAPIを新設。
- `Quel1Box.easy_capture()` のデフォルト動作を設定し、既存設定を変更しないでキャプチャできるようにした。

### v0.7.21
- 年度末リリースの理化学研究所様向け QuEL-1 SE（2-8GHz版)のboxtypeのIDを`quel1se-riken8`に確定。
   - 従来のデバグ用設定は、`x-quel1se-riken8` として残す。
- QuEL-1 SE では、`quel1_linkup` コマンドの動作が、デフォルトで `--use_204c` が付いた状態になる。
   - `--use_204c`無しでリンクアップしたい場合には、実験用コマンドの `quel1_test_linkup`コマンドが使える。

### v0.7.20
- `SimpleBox.start_channel()` の `channel`引数に誤ってデフォルト値が設定されたのを削除。
  - 申し訳有りません。従来動作を維持するためには、`channel=0` を引数に追加してください。 
- いくつかのAPIを拡張しAD9082のDACの`fullscale_current`を変更を可能とした。
  - cssレベルでは新規APIを追加
  - boxレベルでは既存APIに引数を追加
- QuEL-1 SE向けの改修
  - のリンクアップ時にRF Switchを閉じられない不具合を修正。
  - QuEL-1 SE のアナログコンバインポートを `SimpleBox`のAPIで扱うために `subport` の概念を導入。
    - アナログコンバインのない従来のポートには一切の影響無し。
    - 将来的にデジタルコンバインポートも同様に扱う予定。
  - boxレベルのAPIで7.5GHz以下のLO設定が簡単にできるようになった。

### v0.7.19
- リンクアップ関連のログ表示を整理
- ConfigSubsystemのファームウェアインターフェースの改善
  - FPGAとの通信をチューニング・堅牢化
  - ファームウェアのバージョン検出を実装

### v0.7.18
- JESD204Cのリンクアップ周辺を適正化
  - AD9082のキャリブレーションの設定を指定可能にした
  - AD9082のCTLEフィルタの設定周辺の不具合修正
    - 未公開機能だけに影響。この修正の効果を確認後、公開機能に格上げする予定。
    
### v0.7.17
- キャプチャデータの長さが0になる異常ケースの処理を適正化
- 設定系のパケットがたまに通らなくなる問題を解消
- quel1_test_linkup を改修
    - `skip_init` オプションを追加
    - 失敗時の診断を詳細化

### v0.7.16
- 新しいファームウェア simplemulti_20240125 にツール群を適合
  - `quel1_test_linkup` の結果集計を改善 

### v0.7.15
- 新しいファームウェア simplemulti_20240125 に暫定対応
  - e7awgsw が3バージョンに分かれてしまっているのは改善する予定
- quel1_linkup コマンドの初期化エラーを無視するための引数の書式を変更
  - コロン区切りからカンマ区切りに変更

### v0.7.14
- `quel1_test_linkup` を QuEL-1 SE に対応
  
### v0.7.13
- 2023年度末リリース予定のQuEL-1 SE("x-quel1se-riken8")に仮対応（その5の途中）
  - Monitor-in系用のLOの設定用下層APIを追加
  - 調温制御入りファームウェアへの対応

### v0.7.12
- 2023年度末リリース予定のQuEL-1 SE("x-quel1se-riken8")に仮対応（その4）
  - Read-in系の設定・マッピングを適正化

### v0.7.11
- 2023年度末リリース予定のQuEL-1 SE("x-quel1se-riken8")に仮対応（その4）
  - ヒータ制御に対応

### v0.7.10
- 2023年度末リリース予定のQuEL-1 SE("x-quel1se-riken8")に仮対応（その3）
  - 各ボードの温度センサの開発用途向け読み出しに対応

### v0.7.9
- 2023年度末リリース予定のQuEL-1 SE("x-quel1se-riken8")に仮対応（その2）
  - RF Switchに対応

### v0.7.8
- v0.7.7で混入したQuEL-1 SE関連のバグを修正

### v0.7.7
- 2023年度末リリース予定のQuEL-1 SE("x-quel1se-riken8")に仮対応
  - ADDAボードとMixerボードが使用可能
  - RF Switchは未対応
  - 調温系も未対応

### v0.7.6
- ビルド済みパッケージの配布方法を変更。
  - 詳しくは GETTING_STARTED.md を参照のこと。

### v0.7.5
- 2023年度末リリース予定のQuEL-1の変種（"quel1-nec")に仮対応

## v0.7.4 (Public Release)
- `quel1_firmware_version` コマンドを追加。
- ファームウエアバージョン識別子を追加。
  - simplemulti_20231228
  - feedback_20240110（ベータ版）
- ファームウェアのライフサイクル管理に対応。サポート終了・および終了予定のファームウェアについて警告メッセージを出すようにした。
  - feedback_20231108 はサポート対象から除外予定。（当初よりベータ版）

### v0.7.3
- `quel1_dump_config` コマンドの --mxfe オプションを廃止。

### v0.7.2
- ファームウェアのバージョン識別子を追加。
   - simplemulti_20231216 を追加。
- `E7HwType` を `E7FwType` に改名。

### v0.7.1
- QuBEに対応。新しい４つのboxtypeを追加し、{Type-A, Type-B} x {QuBE_OU, QuBE_RIKEN}を指定できるようにした。 
  - qube-ou-a: 最初期のモニタ系がないQuBEのType-A
  - qube-ou-b: 同Type-B
  - qube-riken-a: モニタを追加したQuBEのType-A
  - qube-riken-b: 同Type-B
- `SimpleBox`クラスのAPI名のスペルミスを修正
  - switch を swtich と間違えている箇所があったのを修正。 
- `quel1_linkup`コマンドに、制御装置の各種エラーを無視するための引数を追加。 
  - `--background_noise_threshold`: リンクアップ時にチェックするADCの背景ノイズの最大振幅の上限を変更する。デフォルト値は256。

### v0.7.0
- `Quel1ConfigSubsystem`クラスを、`boxtype` ごとにサブクラス化してライン構成の違いを保持し、実行時チェックを強化。
  - タイプAの機体は`Quel1TypeAConfigSubsystem`クラス、タイプBの機体は`Quel1TypeBConfigSubclass`クラスを使用する必要がある。
    - タイプBではハードウェア的に実装されていない read系の使用を試みた場合に例外を発生するよう、チェックを厳格化。
  - `quel_ic_config_utils.create_box_objects()` を使用してオブジェクトを生成している場合には、このクラス変更がユーザコードに影響を与えないはず。
- `quel_ic_config_utils.create_box_objects` の返り値に rmap を追加した。
  - css, wss, linkupper, box --> css, wss, rmap, linkupper, box とした。
- `quel_ic_config_utils.init_box_with_linkup()` の最初の2つの返り値を、1つのDictに統合した。
  - ユーザコードで、link_ok0, link_ok1 と受けて居た場合には、link_ok で受けて、書く要素を link_ok[0] と link_ok[1] で参照するように変更する必要がある。
- `quel_ic_config_utils.init_box_with_reconnect()` を追加した。再リンクアップ無しでハードウェアを使用するための初期化を行う。boxオブジェクトが存在すれば、box.init() を呼ぶだけなのだが、対応するboxクラスが定義されていないハードウェアを扱うためにこのヘルパー関数を用意した。
- 設定用 json ファイルに、"feature" 条件を追加した。これは、ファームウェアバージョンの自動検出機能と連携して初期化内容を切り替えるのに用いるので、ユーザは直接触ることはない。

### v0.6.8
- 各CLIコマンドに、制御装置の各種エラーを無視するための引数を追加。 
  - `--ignore_crc_error_of_mxfe`: リンクアップ時および再接続時のCRC_ERRORを無視するMxFEを指定。
  - `--ignore_access_faulure_of_adrf6780`: リンクアップ時に疎通確認ができなくてもエラーを発しないADRF6780の個体番号を指定。
  - `--ignore_lock_failure_of_lmx2594`: リンクアップ時にPLLのロックに失敗してもエラーを発しないLMX2594の個体番号を指定。
- feedback版ファームウェア(au50_feedback_20231108)がサポートするread系とmonitor系の同時使用に対応。
  - 将来的には simplemulti版ファームウェアも read系とmonitor系の同時使用をサポートするので、feedback版に固有の機能ではないことに注意。
- ファームウェアバージョンの自動検出を実装。
  - quel_ic_config_utils.create_box_objects() を使用する際に、ファームウェアバージョンの指定は不要。
  - e7awgswパッケージのファームウェアの整合性も自動チェックする。

### v0.6.7
- 設定用 json ファイル中のキー "tx"と"rx"を、それぞれ、"dac"と"adc"に変更した。
  - JESD204C の tx/rx と AD9082のAPI名で使用されている tx/rx は、互いに意味が逆になっている。tx/rxを不用意に使うことで生じる混乱を避けるための名前変更。

## v0.6.6 (Public Release)
- Quel-1 のリンクアップや状態確認用のコマンド群を提供した初の公開バージョン。
  - いまだAPI設計が安定しているとは言えない部分があるが、変更点を提供できる程度には安定化してきているので本文書にて更新点のリストを提供開始する。
  - `quel_ic_config` パッケージの中はかなり安定してきている。
  - 試作的なコードは `quel_ic_config_utils`パッケージに置いてある。`WaveSubsystem`クラス, `SimpleBoxObject`クラス など、設定サブシステムの外側については安定化後に別パッケージを興して分離する予定。分離の際にクラス名称変更の可能性がある。
