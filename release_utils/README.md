# release_utils

リリース作業の補助するツールです。

## `/build.sh`

quelware リポジトリのリリース物として登録する、ビルト済みパッケージアーカイブを作成するためのスクリプトです。

以下の手順でアーカイブをビルドします。

```shell
./build.sh
```

これにより生成される `quelware_prebuild.tgz` を、リリースの追加の際に添付します。

`quel_ic_config/download_prebuilt.sh` で自動ダウンロードするので、ファイル名は変更しないでください。
