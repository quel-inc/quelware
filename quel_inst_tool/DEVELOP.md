# MS2720TのAPI使用時の注意点

## 実験モード
観測しながらスペアナの操作やトレースの取得を行う場合に，Ms2720t.experimental_mode=Trueとすることで連続スイープモードにすることができる。実際にはMs2720t.experimental_modeはMs2720t.init_contと同値である。Ms2720t.experimental_mode = Ms2720t.init_cont = Trueの時に，トレース取得メソッドMs2720t._trace_capture内では，INIT:IMM指令（有限スイープ指令）をスキップするようにしている。また，デストラクタ__del__でMs2720t.init_cont = Trueとし，連続スイープモードに戻るようにした。

## ピーク取得
MS2720tでは，:CALC:MARKer1:MAX指令でカーソルを最大値に移動させたあと，:CALC:MARKer1:MAX:NEXT指令を次々に送ることで，振幅の大きい順で各ピークへカーソルを移動させることができる。これを利用して，カーソル移動毎に:CALC:MARKer1 X(Y)?クエリを送信しピークの周波数と振幅を取得している。また，取得後に周波数の昇順にソートしている。
　見つけたピークの最後になるとカーソルが移動しなくなることを利用して:CALC:MARKer1:MAX:NEXTのループを抜けるようにしているが，念のためにループ回数に上限（Default:10）を設けている。

