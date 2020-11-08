# forecaster

## 実行例

- docker コンテナのビルド&起動

    ```bash
    make up
    ```

- トイデータでのモデルテスト
    - 実行（学習）

        ```bash
        make toy.test
        ```

    - 実行結果確認
        - 学習中に途中経過も確認可能

        ```bash
        make tensorboard
        ```


## 仕様概要(検討中)

- 長期予測ができる
    - 予測対象固有の動きを学習する
        - 決定的な予測で良い
- 短期予測もできる
    - リアルタイム性が高いデータを活用して短期単位の予測精度をあげる
- 長期と短期の組み合わせ
    - 年単位と週単位
    - 日・時間単位と秒単位
    - 予測単位：短期よりも細かい

- 未来時点でも既知のデータ
    - 共通
        - カテゴリ：予測対象の候補キーセット / group keys
        - 数値：イベント情報
    - 長期予測
        - 年周期 ... 季節性に影響をうける時系列
            - カテゴリ：月、日、週番号、曜日
        - 週周期
            - カテゴリ：日、曜日
        - 日周期
            - カテゴリ：時間、分
    - 短期予測
        - なし
- 未来時点では未知のデータ
    - 長期予測
        - 数値：予測対象固有の時系列データ
        - 数値：予測対象の関連時系列データ
        - 年周期の例
            - 未来時点の天候情報
                - 天気、温度、湿度
        - 日周期の例
            - 未来時点の関連情報
                - 音、周辺機器の時系列データ
    - 短期予測
        - 数値：予測対象固有の時系列データ
        - 数値：予測対象の関連時系列データ
            - 周辺環境情報
                - 周辺機器のセンサー情報
                    - 音、振動、赤外線など


- DataSpec
    - Known / exog
        - categorical
            - targets keys
            - day/time relatives
            - event/accident/disaster on/off
        - numeric
            - event/accident/disaster counts
    - Unknown / multivariate
        - categorical
            - weather relative (predictions)
            - event/accident/disaster (predictions)
        - numeric
            - targets
            - targets relatives


