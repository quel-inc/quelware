# パッケージへのファームウェアの追加
各機体固有の情報を含まないオリジナルの bitファイルは、`quel_staging_tool/plain_bits` フォルダに配置する。
サブフォルダの名前は3つのパートから成り、最初のパートが適用対象のFPGAを決定する。現状は、`au50` か `exstickge` の
どちらかである。残りのパートは、人間用なので処理とは関係ないが、バリアントの名前とビルドの日付け、としている。

各サブフォルタの中身は、bitファイルとmmiファイルである。
現状、それぞれの名前は、`top.bit` と　`ram_loc.mmi` で固定している。
新しいフォルダを所定の名前で作成して２つのファイルを配置し、パッケージをリビルドすれば、新しいファームウェアの配布用パッケージを作成できる。

ファームウェアを追加しつづけると、パッケージが肥大化していくので、いずれ、古いファームウェアを削除する必要があるだろう。

# 今後の方向性
意図的に個体管理のDBを参照する機能を除外している。
これらの機能は、別のパッケージでの対応とするつもりである。