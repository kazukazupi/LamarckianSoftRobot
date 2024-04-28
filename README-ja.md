# Lamarckian Soft Robot

## 概要
[Genetic and Evolutionary Computation Conference (GECCO-2024)](http://gecco-2024.sigevo.org/HomePage)に採択された論文「Lamarckian Co-design of Soft Robots via Transfer Learning」の実装のgithubレポジトリです。
深層強化学習による学習結果を個体間で共有することで、より効率の良いソフトロボットの設計を可能にしました。ロボットのシミュレーションには[evogym](https://evolutiongym.github.io/)を用いています。

## 環境構築

### evogymの環境構築
[evogymのgithubレポジトリ](https://github.com/EvolutionGym/evogym)のInstallationに従い、evogymを動かすための環境構築を行なってください。condaを用いた場合でしか検証を行っていないので、condaを用いる子をと推奨します。

### 本レポジトリのための環境構築
このレポジトリをローカルにcloneしてください。

その上でevogym環境に入ります。
```bash
conda activate evogym
```
本レポジトリに特有のモジュールを以下のようにインストールします。

```bash
conda install pydantic
```

## Pull Requestを出す場合
Pull Requestを出す場合は、フォーマッタ・リンターもインストールしてください:

```bash
conda install isort
conda install black
conda install mypy
```

フォーマッタ・リンターをかけてからPull Requestを出してください:
```bash
isort ./
balck ./
mypy ./
```

## チュートリアル
### まずは走らせてみる
以下のコードでロボットを進化させることができる（実行には数分かかる）。
```bash
python ./example/run.py --max-iters 10 --population-size 5 --max-evaluations 10 --exp-dir ./result/experiment
```

### 中断された実験を再開する
アクシデントにより実験が途中で中断されてしまった場合、その実験結果を格納するディレクトリを指定することで実験を再開することができます。
```bash
python ./example/from_middle.py -e <再開する実験のディレクトリ>
```