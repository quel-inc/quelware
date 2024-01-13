# API更新リスト

## v0.7.4
- quel1_firmware_version コマンドを追加。
- ファームウエアバージョン識別子を追加。
  - simplemulti_20231228
  - feedback_20240110（ベータ版）
- ファームウェアのライフサイクル管理に対応。サポート終了・および終了予定のファームウェアについて警告メッセージを出すようにした。
  - feedback_20231108 はサポート対象から除外予定。（当初よりベータ版）

## v0.7.3 (not published)
- quel1_dump_config コマンドの --mxfe オプションを廃止。

## v0.7.2 (not published)
- ファームウェアのバージョン識別子を追加。
   - simplemulti_20231216 を追加。
- E7HwType を E7FwType に改名。

### v0.7.1 (not published)
- QuBEに対応。新しい４つのboxtypeを追加し、{Type-A, Type-B} x {QuBE_OU, QuBE_RIKEN}を指定できるようにした。 
  - qube-ou-a: 最初期のモニタ系がないQuBEのType-A
  - qube-ou-b: 同Type-B
  - qube-riken-a: モニタを追加したQuBEのType-A
  - qube-riken-b: 同Type-B
- SimpleBoxクラスのAPI名のスペルミスを修正
  - switch を swtich と間違えている箇所があったのを修正。 
- quel1_linkupコマンドに、制御装置の各種エラーを無視するための引数を追加。 
  - `--background_noise_threshold`: リンクアップ時にチェックするADCの背景ノイズの最大振幅の上限を変更する。デフォルト値は256。

### v0.7.0 (not published)
- Quel1ConfigSubsystemクラスを、boxtype ごとにサブクラス化してライン構成の違いを保持し、実行時チェックを強化。
  - タイプAの機体はQuel1TypeAConfigSubsystemクラス、タイプBの機体はQuel1TypeBConfigSubclassクラスを使用する必要がある。
    - タイプBではハードウェア的に実装されていない read系の使用を試みた場合に例外を発生するよう、チェックを厳格化。
  - quel_ic_config_utils.create_box_objects() を使用してオブジェクトを生成している場合には、このクラス変更がユーザコードに影響を与えないはず。
- quel_ic_config_utils.create_box_objects の返り値に rmap を追加した。
  - css, wss, linkupper, box --> css, wss, rmap, linkupper, box とした。
- quel_ic_config_utils.init_box_with_linkup() の最初の2つの返り値を、1つのDictに統合した。
  - ユーザコードで、link_ok0, link_ok1 と受けて居た場合には、link_ok で受けて、書く要素を link_ok[0] と link_ok[1] で参照するように変更する必要がある。
- quel_ic_config_utils.init_box_with_reconnect() を追加した。再リンクアップ無しでハードウェアを使用するための初期化を行う。boxオブジェクトが存在すれば、box.init() を呼ぶだけなのだが、対応するboxクラスが定義されていないハードウェアを扱うためにこのヘルパー関数を用意した。
- 設定用 json ファイルに、"feature" 条件を追加した。これは、ファームウェアバージョンの自動検出機能と連携して初期化内容を切り替えるのに用いるので、ユーザは直接触ることはない。

### v0.6.8 (not published)
- 各CLIコマンドに、制御装置の各種エラーを無視するための引数を追加。 
  - `--ignore_crc_error_of_mxfe`: リンクアップ時および再接続時のCRC_ERRORを無視するMxFEを指定。
  - `--ignore_access_faulure_of_adrf6780`: リンクアップ時に疎通確認ができなくてもエラーを発しないADRF6780の個体番号を指定。
  - `--ignore_lock_failure_of_lmx2594`: リンクアップ時にPLLのロックに失敗してもエラーを発しないLMX2594の個体番号を指定。
- feedback版ファームウェア(au50_feedback_20231108)がサポートするread系とmonitor系の同時使用に対応。
  - 将来的には simplemulti版ファームウェアも read系とmonitor系の同時使用をサポートするので、feedback版に固有の機能ではないことに注意。
- ファームウェアバージョンの自動検出を実装。
  - quel_ic_config_utils.create_box_objects() を使用する際に、ファームウェアバージョンの指定は不要。
  - e7awgswパッケージのファームウェアの整合性も自動チェックする。

### v0.6.7 (not published)
- 設定用 json ファイル中のキー "tx"と"rx"を、それぞれ、"dac"と"adc"に変更した。
  - JESD204C の tx/rx と AD9082のAPI名で使用されている tx/rx は、互いに意味が逆になっている。tx/rxを不用意に使うことで生じる混乱を避けるための名前変更。

## v0.6.6
- Quel-1 のリンクアップや状態確認用のコマンド群を提供した初の公開バージョン。
  - いまだAPI設計が安定しているとは言えない部分があるが、変更点を提供できる程度には安定化してきているので本文書にて更新点のリストを提供開始する。
  - quel_ic_config パッケージの中はかなり安定してきている。
  - 試作的なコードは quel_ic_config_utilsパッケージに置いてある。WaveSubsystemクラス, SimpleBoxObjectクラス など、設定サブシステムの外側については安定化後に別パッケージを興して分離する予定。分離の際にクラス名称変更の可能性がある。
